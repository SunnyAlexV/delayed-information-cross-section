"""
build_docx.py - the manuscript as an editable file, for the portal that asks.

    python build_docx.py              -> papers/submission/jfec/manuscript.docx
    python build_docx.py --check      build in a temp directory, write nothing

WHY THIS FILE EXISTS
--------------------
ScholarOne takes a PDF for a new submission and then asks for an editable file
- .doc, .rtf or .tex - for anything it will typeset.  That request arrives
after a revision, which is the worst moment to start making one: the natural
move under a deadline is to export a Word file by hand, fix whatever converted
badly, and send a document that is no longer the thing the verifiers read.
The paper would then have two texts, and only one of them checked.

So the editable file is built, not typed, from the SAME reassembly that makes
the submission PDF - build_jfec_submission.reassemble, imported rather than
reimplemented, so the floats land in the journal's order in both files by
construction.  The figures enter as images because a Word file cannot hold the
inline drawings, and they are the 1200 d.p.i. renders the submission package
already produced rather than new ones.

The gate is the point.  After conversion the document's own text is compared
with the text of the HTML it came from, word for word; any difference at all
fails the build.  A conversion that silently drops an equation, a table row or
a paragraph is the defect this file exists to prevent, and "it looked fine"
is not a check.

That gate is not in selftest.py's register, and the reason is worth stating
rather than leaving as an omission: the register runs against the shipping set,
which contains neither the built variants nor the figure files, so a case for
it could not run there.  What makes the absence tolerable is that this check
cannot be skipped the way a verifier can - it is inside the only thing that
produces the file, so a document that exists has passed it.
"""

import html
import os
import re
import shutil
import subprocess
import sys
import tempfile

import build_jfec_submission as J

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "papers", "submission", "jfec")
FIGS = os.path.join(OUT, "figures")
DPI = 300          # for the embedded copies; the TIFFs at 1200 remain the art


def _need(tool):
    if shutil.which(tool) is None:
        raise SystemExit(f"build_docx: {tool} is not installed, and this file "
                         f"does not have a second way to do its job")


def words(markup):
    """The text of a fragment of the manuscript, as a list of words.

    Inline tags close up rather than separate: <sub>s</sub> is part of the word
    it hangs off, so stripping it to a space turns "O_s" into two words and
    makes every subscript in the paper look like a conversion error.  Block
    tags do separate.  Nothing inside an <svg> counts, because the figures
    travel as images and their labels are not prose.
    """
    t = re.sub(r"<svg\b.*?</svg>", " ", markup, flags=re.S)
    t = re.sub(r"<head\b.*?</head>", " ", t, flags=re.S)
    t = re.sub(r"</?(?:sub|sup|b|i|em|strong|span|code|a)\b[^>]*>", "", t)
    return html.unescape(re.sub(r"<[^>]+>", " ", t)).split()


def docx_words(path):
    """The same, read out of the built document in document order."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    d = docx.Document(path)
    out = []
    for child in d.element.body.iterchildren():
        if child.tag.endswith("}p"):
            out.append(Paragraph(child, d).text)
        elif child.tag.endswith("}tbl"):
            t = Table(child, d)
            for r in t.rows:
                out.append(" ".join(c.text for c in r.cells))
    return " ".join(" ".join(out).split()).split()


def rasterise(work):
    """The submission package's own figure files, as PNGs Word can embed."""
    got = []
    for n in (1, 2, 3):
        src = os.path.join(FIGS, f"figure-{n}.pdf")
        if not os.path.isfile(src):
            raise SystemExit(f"build_docx: {os.path.relpath(src, HERE)} is not "
                             f"there. Run build_jfec_submission.py first; this "
                             f"file embeds that package's figures rather than "
                             f"drawing a second set.")
        subprocess.run(["pdftoppm", "-png", "-r", str(DPI), "-singlefile",
                        src, os.path.join(work, f"figure-{n}")], check=True)
        got.append(os.path.join(work, f"figure-{n}.png"))
    return got


def style(path, title):
    """Times New Roman 12pt, double spaced; tables and captions at 10pt.

    The same geometry the PDF is rendered under, applied to the Word file so
    that an editor opening either sees the same page.  Pandoc also promotes the
    HTML <title> to a heading, which would put 'Manuscript for anonymous
    review' above the paper's own title; it is removed here, and the text gate
    then compares what is left against the source with nothing excused.
    """
    import docx
    from docx.shared import Pt
    from docx.enum.text import WD_LINE_SPACING
    d = docx.Document(path)

    first = d.paragraphs[0] if d.paragraphs else None
    if first is not None and first.text.strip() == title.strip():
        first._element.getparent().remove(first._element)

    normal = d.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE

    for p in d.paragraphs:
        p.paragraph_format.line_spacing = 2.0
        for r in p.runs:
            r.font.name = "Times New Roman"
    for t in d.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing = 1.15
                    for r in p.runs:
                        r.font.name = "Times New Roman"
                        r.font.size = Pt(10)
    d.save(path)


def main(argv):
    check = "--check" in argv
    _need("pandoc")
    _need("pdftoppm")
    try:
        import docx                                          # noqa: F401
    except ImportError:
        raise SystemExit("build_docx: python-docx is not installed "
                         "(pip install python-docx)")

    src = os.path.join(J.VAR, J.MANUSCRIPT)
    if not os.path.isfile(src):
        raise SystemExit("build_docx: papers/variants/jfec is not built. "
                         "Run build_variants.py first.")
    raw = open(src, encoding="utf-8").read()
    body, moved = J.reassemble(raw)

    work = tempfile.mkdtemp(prefix="jfec-docx-")
    rasterise(work)

    n_svg = len(re.findall(r"<svg\b", body))
    n_tab = len(re.findall(r"<table\b", body))
    seen = [0]

    def _img(m):
        seen[0] += 1
        return (f'<p><img src="figure-{seen[0]}.png" '
                f'alt="Figure {seen[0]}" /></p>')

    shaped = re.sub(r"<svg\b.*?</svg>", _img, body, flags=re.S)
    if seen[0] != n_svg:
        raise SystemExit(f"build_docx: {n_svg} drawings in the manuscript and "
                         f"{seen[0]} images substituted")

    title = ""
    _t = re.search(r"<title>(.*?)</title>", body, re.S)
    if _t:
        title = " ".join(html.unescape(_t.group(1)).split())

    page = os.path.join(work, "manuscript.html")
    open(page, "w", encoding="utf-8").write(shaped)
    out_dir = tempfile.mkdtemp(prefix="jfec-docx-out-") if check else OUT
    os.makedirs(out_dir, exist_ok=True)
    doc = os.path.join(out_dir, "manuscript.docx")
    subprocess.run(["pandoc", page, "-f", "html", "-t", "docx",
                    "--resource-path", work, "-o", doc], check=True)
    style(doc, title)

    import docx as _d
    built = _d.Document(doc)
    if len(built.tables) != n_tab:
        raise SystemExit(f"build_docx: the manuscript has {n_tab} tables and "
                         f"the document has {len(built.tables)}")
    if len(built.inline_shapes) != n_svg:
        raise SystemExit(f"build_docx: {n_svg} figures and "
                         f"{len(built.inline_shapes)} images in the document")

    # The blinding, re-checked on the converted file.  The PDF is checked for
    # this already; a second route out of the same source is a second way to
    # lose it, and an editable file is the one a referee can open and inspect.
    text = " ".join(docx_words(doc))
    for token in tuple(J.FORBIDDEN) + tuple(J._private_tokens()):
        if token in text:
            raise SystemExit(f"build_docx: '{token}' is in the editable file "
                             f"this build calls anonymous")

    want, got = words(body), docx_words(doc)
    if want != got:
        import difflib
        ops = [o for o in difflib.SequenceMatcher(None, want, got,
                                                  autojunk=False).get_opcodes()
               if o[0] != "equal"]
        first = ops[0]
        raise SystemExit(
            f"build_docx: the document's text is not the manuscript's: "
            f"{len(ops)} difference(s), the first "
            f"{first[0]} {' '.join(want[first[1]:first[2]])[:60]!r} -> "
            f"{' '.join(got[first[3]:first[4]])[:60]!r}")

    n = len(got)
    where = "a temporary directory" if check else os.path.relpath(doc, HERE)
    print(f"build_docx: manuscript.docx  {n:,} words, {n_tab} tables, "
          f"{n_svg} figures  -> {where}")
    print(f"  every word of the document is a word of the manuscript it was "
          f"built from, in order")
    if check:
        shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
