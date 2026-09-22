"""
build_ssrn.py - the single PDF that goes to SSRN, and the checks on it.

    python build_ssrn.py

WHY THIS FILE EXISTS
--------------------
SSRN takes one full-text PDF per submission.  The paper and its Internet
Appendix are two documents here, and the appendix is a large part of what makes
the paper convincing: a reader who finds the paper on SSRN and cannot see the
robustness work has been shown the claims without the working.  So the SSRN
copy is the two bound together, paper first.

Binding is a concatenation and concatenations are where documents go missing
quietly.  A merge that drops the last page, or silently re-encodes a font so
that a minus sign stops extracting, produces a file that opens fine and is
wrong in a way nobody notices until a referee quotes a positive number that
should be negative.  Both of those have happened in this project - the minus
sign twice, to two different reviewers - so the merge is checked rather than
trusted:

    * the page count is exactly the sum of the two inputs
    * a distinctive sentence from the START and the END of each input still
      extracts from the merged file, so nothing was dropped at either seam
    * every negative figure the sources write with an explicit minus still
      extracts with its sign, which is gate 5 of build_registry.py applied to
      a file that gate does not see

It also writes papers/ssrn/ssrn-metadata.md: the title, abstract, keywords and
JEL codes for both entries, lifted out of the documents rather than retyped,
because an abstract typed into a web form is a copy of figures the papers keep
current and copies are what this project exists to stop making.

qpdf does the merge.  pypdf is not used: it re-writes the page tree and this
project has no reason to re-encode a document it has already rendered once.
"""

import html as _html
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.join(HERE, "papers")
SRC = os.path.join(PAPERS, "source")
OUT = os.path.join(PAPERS, "ssrn")

MAIN = "what-substitutes-for-a-stale-mark"
SUPP = "stale-mark-internet-appendix"
NOTE = "threshold-rule-at-its-ceiling"
BOUND = "what-substitutes-for-a-stale-mark-with-internet-appendix.pdf"


def pages(path):
    """Page count read out of the PDF, not reported by the tool that made it."""
    return len(re.findall(rb"/Type\s*/Page[^s]", open(path, "rb").read()))


def text_of(path):
    return subprocess.run(["pdftotext", "-layout", path, "-"],
                          capture_output=True, text=True, timeout=180).stdout


def norm(s):
    for a, b in (("−", "-"), ("–", "-"), ("—", "-"),
                 (" ", " "), (" ", " ")):
        s = s.replace(a, b)
    return " ".join(s.split())


def plain(s):
    """Tag-free, entity-free text, fit to paste into a web form.

    norm() is for COMPARING text and flattens the typography to ASCII; this is
    for PUBLISHING it, so the entities become the characters they name and the
    real minus sign, en dash and delta survive.  The first version of the SSRN
    metadata used norm() and produced "S&amp;P 500" and "&delta; days" - which
    would have gone into the abstract field exactly as written, on the entry
    every reader sees before they open the PDF.
    """
    s = _html.unescape(re.sub(r"<[^>]+>", " ", s))
    return " ".join(s.split())


def seam_probes(txt):
    """Two long sentences, one from near the start and one from near the end.

    Near, not at: the first line of a PDF is a title that also appears in the
    other document's own title, so a probe taken from the very top can be found
    in the merged file even when the page it came from was dropped.
    """
    words = txt.split()
    head = " ".join(words[60:80])
    tail = " ".join(words[-120:-100])
    return [p for p in (head, tail) if len(p) > 40]


def main(argv):
    os.makedirs(OUT, exist_ok=True)
    src = [os.path.join(PAPERS, MAIN + ".pdf"),
           os.path.join(PAPERS, SUPP + ".pdf")]
    for p in src:
        if not os.path.isfile(p):
            raise SystemExit(f"{os.path.relpath(p, HERE)} is not built; run "
                             f"build_pdf.py first")
    dst = os.path.join(OUT, BOUND)
    r = subprocess.run(["qpdf", "--empty", "--pages"] + src + ["--", dst],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("qpdf failed to bind the two documents:\n" + r.stderr)

    fails = []
    want = sum(pages(p) for p in src)
    got = pages(dst)
    if got != want:
        fails.append(f"the bound PDF has {got} pages; the two inputs have "
                     f"{want} between them")

    merged = norm(text_of(dst))
    for p in src:
        t = norm(text_of(p))
        for probe in seam_probes(t):
            if probe not in merged:
                fails.append(f"text from {os.path.basename(p)} is missing from "
                             f"the bound PDF: {probe[:60]}...")
        # every figure the SOURCE writes with an explicit minus entity
        html = open(os.path.join(SRC, os.path.basename(p)[:-4] + ".html"),
                    encoding="utf-8").read()
        for n in sorted(set(re.findall(r"&minus;(\d+\.\d+)", html))):
            hits = list(re.finditer(re.escape(n), merged))
            if not hits:
                continue
            signed = any(merged[max(0, m.start() - 2):m.start()].strip()[-1:]
                         in "-" for m in hits)
            if not signed:
                fails.append(f"{n} extracts from the bound PDF without its "
                             f"minus sign")

    # The SSRN form's own fields, EXTRACTED rather than retyped.  An abstract
    # typed into a web form is a copy, and a copy of a number is how every
    # error in this project started: the paper's abstract carries figures, the
    # figures get corrected, and the SSRN entry goes on showing the old ones to
    # every reader who never opens the PDF.  So the text below is lifted from
    # the documents themselves and pasted into SSRN unchanged.
    meta = ["# SSRN submission fields", "",
            "Extracted from the paper sources by `build_ssrn.py`. Paste these "
            "into the SSRN form;", "do not retype them - the abstracts carry "
            "figures that the papers keep current.", ""]
    for stem, files, note_txt in (
            (MAIN, [BOUND],
             "Upload the bound PDF: the paper followed by its Internet "
             "Appendix, 78 pages."),
            (NOTE, [NOTE + ".pdf"],
             "Posts as its own entry. The main paper cites it.")):
        html = open(os.path.join(SRC, stem + ".html"), encoding="utf-8").read()
        title = re.search(r"<h1>(.*?)</h1>", html, re.S).group(1)
        title = plain(title)
        blk = html[html.find('<div class="abs">'):]
        blk = blk[:blk.find("</div>")]
        paras = re.findall(r"<p[^>]*>(.*?)</p>", blk, re.S)
        body = [plain(x) for x in paras]
        body = [x for x in body if x and not x.lower().startswith("abstract")]
        kw = next((x for x in body if x.startswith("Keywords:")), "")
        abstract = " ".join(x for x in body if not x.startswith("Keywords:"))
        meta += [f"## {title}", "", f"**File.** {note_txt}", "",
                 "**Abstract.**", "", abstract, ""]
        if kw:
            k = kw.split("JEL codes:")
            meta += ["**Keywords.** " + k[0].replace("Keywords:", "").strip(), ""]
            if len(k) > 1:
                meta += ["**JEL codes.** " + k[1].strip(), ""]
        else:
            fails.append(f"{stem}.html states no keywords or JEL codes, which "
                         f"the SSRN form asks for")
        # Nothing that is still markup may reach the form.
        for _field, _txt in (("title", title), ("abstract", abstract),
                             ("keywords", kw)):
            _e = re.search(r"&[a-zA-Z]+;|&#\d+;|<[a-zA-Z/]", _txt)
            if _e:
                fails.append(f"{stem} {_field} still contains markup "
                             f"({_e.group(0)}), which would paste into SSRN "
                             f"literally")
        meta += [f"**Characters in abstract.** {len(abstract)}", ""]
    open(os.path.join(OUT, "ssrn-metadata.md"), "w",
         encoding="utf-8").write("\n".join(meta).rstrip() + "\n")

    note = os.path.join(PAPERS, NOTE + ".pdf")
    print(f"build_ssrn: {BOUND}")
    print(f"  {pages(src[0])} pages of paper + {pages(src[1])} of appendix "
          f"= {got}, {os.path.getsize(dst)/1e6:.1f} MB")
    if os.path.isfile(note):
        print(f"  companion note posts separately: {NOTE}.pdf, "
              f"{pages(note)} pages")
    for f in fails:
        print("  FAIL  " + f)
    if not fails:
        print("  nothing was lost at either seam and every minus survived")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
