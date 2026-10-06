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

import collections
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


# A proper noun as the letters write them: a capitalised word, optionally
# followed by a second capitalised word or a code like KSE-100.  Document
# furniture is ignored - every section is in an Internet Appendix, and a
# manuscript section number is not a claim about the appendix at all.
PROPER = re.compile(r"\b[A-Z][A-Za-z]+(?:[- ][A-Z0-9][A-Za-z0-9]*)?\b|\b[A-Z]{2,}-\d+\b")
XREF_IGNORE = {"Section", "Sections", "Internet", "Appendix", "Table", "Tables",
               "Figure", "Figures", "The", "It", "That", "This", "Each", "What",
               "For", "Both", "If", "In", "Under", "On", "No", "All", "My",
               "Between", "Across", "Where", "Which", "A", "I", "Of", "One",
               "Two", "Thirty", "Dear", "Editors", "Request", "Yours"}


def appendix_sections(path):
    """{'S38': its text}, read out of the appendix the letter encloses."""
    if not os.path.isfile(path):
        return {}
    t = open(path, encoding="utf-8").read()
    hs = list(re.finditer(r"<h2[^>]*>\s*(S\d+)\.", t))
    out = {}
    for i, m in enumerate(hs):
        end = hs[i + 1].start() if i + 1 < len(hs) else len(t)
        out[m.group(1)] = " ".join(re.sub(r"<[^>]+>", " ", t[m.start():end]).split())
    return out


def checks_for(variant):
    """The verifier's own count, taken from the verifier rather than recalled."""
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "verify_paper.py"),
         os.path.join(VAR, variant, PAPER + ".html"),
         os.path.join(HERE, "papers", "source", "why-a-simple-threshold-rule-is-hard-to-beat.html")],
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

        # The title the letter names must be the title on the manuscript it
        # encloses.  This check was absent for the whole of the submission's
        # preparation, and the absence cost exactly what the file's own preamble
        # predicts: the JFEc manuscript was retitled, the letter was not, and the
        # package was one upload away from telling an editor it enclosed a paper
        # whose title appeared nowhere in it.  Every other fact here is
        # recomputed rather than trusted; the title was the one copied string
        # nobody compared.  Read out of the variant, not from a list of expected
        # titles - a list is a second copy of the thing being checked, and this
        # project has already published one of those.
        _h1 = re.search(r"<h1>(.*?)</h1>", body, re.S)
        if not _h1:
            fails.append(f"{fn}: the variant it describes has no <h1> title")
        else:
            want = " ".join(re.sub(r"<[^>]+>", " ", _h1.group(1)).split())
            _q = re.search(r'I am submitting\s+"(.*?)"', txt, re.S)
            if not _q:
                fails.append(f"{fn}: does not name the title it is submitting")
            else:
                got = " ".join(_q.group(1).split())
                if got != want:
                    fails.append(f"{fn}: names {got!r}; the manuscript is "
                                 f"titled {want!r}")

        # and the appendix must carry that same title, which is the defect this
        # round actually shipped: build_variants retitled the manuscript alone,
        # so the appendix went on announcing itself as belonging to the paper's
        # previous name.  Checked here rather than in verify_paper because it is
        # a property of the PAIR, and a check that reads one document can never
        # see a disagreement between two.
        _ap = os.path.join(VAR, key, "stale-mark-internet-appendix.html")
        if _h1 and os.path.isfile(_ap):
            _ah = re.search(r"<h1>(.*?)</h1>",
                            open(_ap, encoding="utf-8").read(), re.S)
            if not _ah:
                fails.append(f"{fn}: the appendix for {key} has no <h1> title")
            else:
                aps = " ".join(re.sub(r"<[^>]+>", " ", _ah.group(1)).split())
                if want not in aps:
                    fails.append(f"{fn}: the appendix is headed {aps!r}, which "
                                 f"does not carry the manuscript's title "
                                 f"{want!r}")

        # Every appendix section a letter cites must be the section that
        # carries the thing the letter says is there.  The letter that shipped
        # this round cited S39 for the withdrawn series; S39 exists, and is
        # about the volatility proxy, so an editor following the pointer would
        # have found an unrelated section and the only signal would have been
        # his own confusion.  A check that the section EXISTS would have passed.
        # So the subject is read out of the citing sentence instead: the proper
        # nouns it names, kept only where they are distinctive in the appendix,
        # must appear in the section it sends the reader to.  Distinctive means
        # the token occurs in at most a quarter of the sections - a word in half
        # of them identifies nothing, and demanding it would cry wolf, which
        # this file's other comment explains is worse than not checking.
        _ap_sec = appendix_sections(_ap)
        if _ap_sec:
            _df = collections.Counter()
            for _b in _ap_sec.values():
                for _t in set(PROPER.findall(_b)):
                    _df[_t] += 1
            _cap = len(_ap_sec) * 0.25
            _sents = re.split(r"(?<=[.!?])\s+", " ".join(txt.split()))
            for _i, _s in enumerate(_sents):
                for _m in re.finditer(r"Section (S\d+)\b", _s):
                    _sid = _m.group(1)
                    if _sid not in _ap_sec:
                        fails.append(f"{fn}: cites {_sid}, which the appendix "
                                     f"it encloses does not have")
                        continue
                    _ctx = (_sents[_i - 1] + " " if _i else "") + _s
                    # A pointer into the MANUSCRIPT is not a claim about the
                    # appendix, so it is removed before the subject is read.
                    _ctx = re.sub(r"\bSections?\s+S?\d+[\d,\sSand]*", " ", _ctx)
                    _want = {_t for _t in PROPER.findall(_ctx)
                             if _t not in XREF_IGNORE and 0 < _df[_t] <= _cap}
                    _gone = sorted(_t for _t in _want if _t not in _ap_sec[_sid])
                    if _gone:
                        fails.append(f"{fn}: cites {_sid} for {', '.join(_gone)}, "
                                     f"which that section does not mention")

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
                # Two shapes, because the letters use both: "N pages" and the
                # second figure written as "and N if every line ... double
                # spaced".  Only the first was read for several revisions, so
                # two of the three letters carried a second page count nobody
                # checked - and both were SMALLER than the first, which is
                # arithmetically impossible: double spacing every line cannot
                # shorten the manuscript.  That is the same class of error as
                # the 95% [47, 75] pair, found the same way, by asking what the
                # checker was not reading rather than whether it passed.
                stated = set(re.findall(r"(\d+) pages", txt)) | \
                         set(re.findall(r"is (\d+) pages", txt)) | \
                         set(re.findall(r"and (\d+) if every line", txt))
                stated = {int(x) for x in stated}
                # A journal whose own order puts every float on a page of its
                # own produces a THIRD count, larger than both of these and just
                # as real.  build_jfec_submission.py measures it and records it
                # in MEASURED.tsv, so a letter may state it - and only if the
                # file agrees.  Admitting it by rule rather than by exception
                # keeps the check exact: a number that is in neither the two
                # geometries nor the recorded package still fails.
                _mt = os.path.join(HERE, "papers", "submission", key,
                                   "MEASURED.tsv")
                _pkg = {}
                if os.path.isfile(_mt):
                    for _ln in open(_mt, encoding="utf-8"):
                        _k, _, _v = _ln.partition("\t")
                        if _v.strip().isdigit():
                            _pkg[_k.strip()] = int(_v)
                admissible = {floors, strict} | set(_pkg.values())
                # A letter may describe either object - the variant, in its two
                # geometries, or the assembled package, in text-and-references
                # against the whole file - but it must quote a COMPLETE pair for
                # whichever it describes.  Quoting one number of a pair is how a
                # letter states the flattering figure and omits the other; this
                # check existed for that and is kept.  What it used to assume is
                # that only the variant could be described, so a letter quoting
                # the package's own 40 and 50 - the numbers for the file the
                # editor actually receives - failed for being accurate about a
                # different document.
                _pairs = [{floors, strict}]
                if {"text_and_references_pages", "manuscript_pages"} <= set(_pkg):
                    _pairs.append({_pkg["text_and_references_pages"],
                                   _pkg["manuscript_pages"]})
                if stated and not stated <= admissible:
                    fails.append(f"{fn}: states pages {sorted(stated)}; measured "
                                 f"{floors} (floats single) and {strict} (all "
                                 f"double), package {sorted(set(_pkg.values()))}")
                elif stated and not any(_pr <= stated for _pr in _pairs):
                    fails.append(f"{fn}: states only page count(s) {sorted(stated)}, "
                                 f"which is part of a pair and not a pair. Quote "
                                 f"either {sorted(_pairs[0])} for the variant's two "
                                 f"geometries"
                                 + (f" or {sorted(_pairs[1])} for the assembled "
                                    f"package's text-and-references against its "
                                    f"whole file" if len(_pairs) > 1 else "")
                                 + ", not one number from one of them")
                if floors > strict:
                    fails.append(f"{fn}: measured {floors} pages with floats single "
                                 f"and {strict} with every line double, which cannot "
                                 f"be - the stricter setting cannot be shorter")

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
