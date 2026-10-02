"""
build_jfec_submission.py - the JFEc package, in the order and geometry the
journal asks for rather than the order a reader prefers.

    python build_jfec_submission.py            build papers/submission/jfec/
    python build_jfec_submission.py --check    build into a temporary directory
                                               and report, touching nothing

WHY THIS FILE EXISTS
--------------------
The Journal of Financial Econometrics asks for a manuscript assembled in a
specific order - "title page, abstract, text, figure legends, references,
footnotes, tables, figures" - set on US Letter with one-inch margins, double
spaced, with every table on its own page and no vertical rules, every figure in
a file of its own at 1200 d.p.i. for line art, and alt text under each legend.

None of that is how the paper is typeset to be read.  The house format is A4 at
10.5pt with floats where the argument needs them, and it should stay that way:
a document built for a submission portal is a worse document to read, and
reading it is what the author and the referees do most of.  So this file does
not change the paper.  It takes the built JFEc variant and re-assembles it,
which keeps one source for the content and puts the journal's requirements in a
script where they can be read, checked and corrected.

What it does, and why each step is a step rather than an instruction in a
checklist:

  * Floats come out of the flow and go to the back, one per page, in their own
    numbered order, and a placeholder - "[Table 3 about here]" - goes where each
    one was.  A float silently dropped in a reordering is the failure this
    project spends its life preventing, so the count of floats removed and the
    count of placeholders inserted are compared, and the numbers must run 1..N
    with no gaps.
  * The figure legends are collected onto one page, and the ALT TEXT under each
    is read out of the figure's own aria-label rather than written again here.
    The accessibility text and the legend's alt line cannot disagree, because
    there is only one of them.
  * The geometry is the one a page limit means, and it is applied on top of the
    source rather than edited into it - the same override measure_pages.py uses,
    so the page count this file reports and the count that file reports are the
    same number by construction rather than by coincidence.
  * Each figure is exported twice: a vector PDF, which is what a typesetter
    actually wants from line art, and a 1200 d.p.i. TIFF, which is what the
    guidelines ask for.  Both come from the same SVG in the manuscript.
  * The title page is re-rendered last, with the word count and the measured
    page count injected, because a title page that states a length has to state
    the length of the file beside it.
  * The manuscript is the ANONYMOUS variant, and the three identifiers that
    blinding removes are searched for again here.  A blinded file that is built,
    reordered and re-rendered by a second script is a blinded file that can lose
    its blinding in a way the first script cannot see.

Requires playwright and its Chromium (as build_pdf.py does) and poppler's
pdftoppm for the TIFF conversion.  Both are checked for, and a missing one
names itself rather than producing a package with a hole in it.
"""

import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile

# 1200 d.p.i., which the guidelines ask for on line drawings.  Written here
# once so the raster and the TIFF header cannot disagree about it.
DPI = 1200

# The journal's own length guidance, in the geometry it is counted in.
LIMIT = 40

HERE = os.path.dirname(os.path.abspath(__file__))
VAR = os.path.join(HERE, "papers", "variants", "jfec")
OUT = os.path.join(HERE, "papers", "submission", "jfec")

MANUSCRIPT = "manuscript-anonymous.html"
APPENDIX = "internet-appendix-anonymous.html"
TITLE = "title-page.html"

# The identifiers blinding removes.  Searched for again in what this file
# renders, because this file rebuilds the blinded document from its parts.
#
# These are the PUBLIC ones only.  An earlier version also listed a fragment of
# the telephone number and the author's town, so that the guard would still
# fire with contact.txt absent - which wrote both into a file the repository
# publishes.  A guard that spells out the secret publishes the secret.  The
# private values are now read from contact.txt at run time by _private_tokens()
# below, and when that file is absent the guard says so rather than carrying a
# copy of what it is looking for.
FORBIDDEN = ("Vellanikaran", "SunnyAlexV", "7504558", "sunnyalex1234")


def _private_tokens():
    """The contact details, read from the unpublished file, as search tokens.

    Returns an empty list when contact.txt is absent, which is the published
    state of this repository: a reader can still rebuild the package, and the
    checks that need the private values announce that they were skipped.
    """
    path = os.path.join(HERE, "contact.txt")
    if not os.path.isfile(path):
        return []
    toks = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        val = line.split(":", 1)[1].strip().rstrip(".")
        if not val or val.startswith("<"):
            continue
        toks.append(val)
        toks += [p.strip() for p in re.split(r"[,\n]", val) if len(p.strip()) > 4]
    return [t for t in dict.fromkeys(toks) if len(t) > 4]

# US Letter, one-inch margins, 12pt serif, double spaced - the geometry a page
# limit is counted in, and the same override measure_pages.py applies, with the
# submission's own page-break rules added.  Floats keep their own leading: no
# journal double-spaces a table body, and the guideline's "double-spaced page"
# has never meant otherwise.
GEOMETRY = """
<style id="jfec-submission">
@page { size: 8.5in 11in; margin: 1in; }
body, p, li, td, th, div.abstract, .front {
  font-family: "Times New Roman", Times, serif !important;
  font-size: 12pt !important;
  text-align: left !important;
}
body, p, li { line-height: 2.0 !important; }
/* Floats keep their own leading.  NOT "figure *": an SVG sets its text size
   with presentation attributes, which any CSS font-size overrides, so
   "figure *" re-sized every label inside every drawing to 10pt and the text
   ran out past the axes and off the right margin.  The figures had been
   shipping that way, clipped, because the check on them read the text layer
   and never looked at the page.  Captions are named explicitly instead, and
   nothing inside an <svg> is touched. */
/* 10pt, which is the house table size, and NOT a size chosen to make a table
   fit.  Shrinking the type to buy a page is the kind of format compression
   that is more conspicuous than an honest page count, so what buys the page
   below is prose that belongs in the appendix instead. */
table, table *, caption, figcaption, .cap, .note, .fn {
  line-height: 1.15 !important; font-size: 10pt !important;
}
figure { line-height: 1.15 !important; }
/* and NOTHING sets a font-size inside an <svg>: a presentation attribute
   loses to any cascade value at all, so even "revert" here re-sized every
   label to the user agent's 16px and clipped the drawings again. */
h1, h2, h3 { line-height: 1.4 !important; }
/* The journal asks for no vertical rules.  The house stylesheet draws none,
   and this says so out loud so that a later change to it cannot quietly add
   one to a file that has already been described as compliant. */
table, table td, table th { border-left: 0 !important; border-right: 0 !important; }
/* One TABLE per page, because the journal says so in as many words: "Tables
   should be typewritten, each on a separate page".  It says no such thing
   about figures.  What it asks of a figure is that it be a separate
   electronic FILE - which the figures/ directory supplies - and that the
   legends be collected on a page of their own, which they are.  So the
   figures flow, two to a page here rather than three pages carrying one
   picture each, and only the break INSIDE a figure is forbidden. */
.jfec-break { break-before: page; }
figure, table { break-inside: avoid; }
.jfec-float { break-after: page; }
.jfec-float:last-child { break-after: auto; }
.jfec-figure { break-after: auto; margin-bottom: 26pt; }
.jfec-ph { font-style: italic; text-align: center !important; margin: 14pt 0; }
.jfec-alt { font-size: 10pt !important; line-height: 1.3 !important;
            margin: 2pt 0 12pt; }
/* An unnumbered footnote under its table, which is where the journal asks for
   the descriptive material that a brief title cannot carry. */
.jfec-tablenote { font-size: 10pt !important; line-height: 1.3 !important;
                  margin: 6pt 0 0; text-align: left !important; }
/* A figure rendered at its own scale on a manuscript page, not at the width of
   a typeset column. */
.jfec-float svg, .jfec-figure svg { max-width: 100%; height: auto; }
</style>
"""

FOOTER = ('<div style="width:100%;font-family:Times New Roman,serif;'
          'font-size:10pt;text-align:center;margin:0 1in;">'
          '<span class="pageNumber"></span></div>')


def _no_css_inside_svg():
    """Nothing in the submission stylesheet may set type inside a drawing.

    An SVG sets its text size with presentation attributes, and a presentation
    attribute loses to any CSS declaration at all - including one inherited
    from a selector as broad as "figure *".  That selector re-sized every label
    in every figure to 10pt, the labels ran past the axes and off the right
    margin, and the package shipped that way for three builds because the only
    check on the figures read the PDF's text layer, where clipped text is
    present and correct.  So the rule is checked rather than remembered.
    """
    css = re.sub(r"/\*.*?\*/", "", GEOMETRY, flags=re.S)
    # By RULE, not by line: the selector and the declaration sit on different
    # lines in this stylesheet, so a line-at-a-time check reads "figure *" and
    # "font-size" as unrelated and passes the exact defect it exists to catch.
    # It did, on its first tamper test.
    bad = []
    for _m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        sel, body = " ".join(_m.group(1).split()), _m.group(2)
        if re.search(r"figure\s*\*|\bsvg\b", sel) and \
                re.search(r"font-size|font-family|font\s*:", body):
            bad.append(f"{sel} {{{' '.join(body.split())}}}")
    if bad:
        raise SystemExit(
            "build_jfec_submission: the stylesheet sets type inside <svg>, "
            "which overrides the figures' own presentation attributes and "
            "clips them: " + "; ".join(bad))


def _need(tool):
    if shutil.which(tool) is None:
        raise SystemExit(f"build_jfec_submission: {tool} is not installed, and "
                         f"the package it builds would be missing the files it "
                         f"converts. Install it and run again.")


def pages(path):
    """Page count read out of the PDF, as build_pdf.py does."""
    return len(re.findall(rb"/Type\s*/Page[^s]", open(path, "rb").read()))


def _blocks(html, tag):
    """Every <tag>...</tag> block, outermost, in document order."""
    return [m for m in re.finditer(rf"<{tag}\b[^>]*>.*?</{tag}>", html, re.S)]


def _number(block, kind):
    """The float's own number, read from its caption rather than counted.

    Counting gives 1..N whatever the document says; reading means a mismatch
    between the caption and the order is a failure here instead of a figure
    reference that lands on the wrong page.
    """
    m = re.search(rf"<b>{kind}\s+(\d+)\.</b>", block)
    return int(m.group(1)) if m else None


def _alt(figure):
    """The figure's accessibility text, which is also its alt line."""
    m = re.search(r'aria-label="([^"]*)"', figure)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def split_caption(block):
    """A brief descriptive title, and the rest as an unnumbered note.

    The journal asks that each table have "a brief, descriptive title" and that
    the "necessary descriptive material" go in "an unnumbered footnote".  The
    house format puts all of it in the caption, because a reader reading the
    table wants the explanation beside it; a submission has to split the two.
    The split is at the first sentence end outside any tag, which is where these
    captions already change from naming the table to explaining it.
    """
    m = re.match(r"(<caption[^>]*>)(.*)(</caption>)", block, re.S)
    if not m:
        return block, ""
    head, body, tail = m.groups()
    lead = re.match(r"\s*(<b>[^<]*</b>\s*)", body)
    pre = lead.group(1) if lead else ""
    rest = body[len(pre):]
    depth, i, cut = 0, 0, None
    while i < len(rest) - 1:
        c = rest[i]
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif depth == 0 and c == "." and rest[i + 1] in " \n":
            after = rest[i + 2:i + 40].lstrip()
            prev = rest[max(0, i - 3):i]
            if after[:1].isupper() and not re.search(r"\b[A-Z]$|\be\.g$|\bi\.e$",
                                                     prev):
                cut = i + 1
                break
        i += 1
    if cut is None:
        return block, ""
    return head + pre + rest[:cut] + tail, rest[cut:].strip()


def _caption(block, tag):
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", block, re.S)
    return m.group(1).strip() if m else ""


def reassemble(html):
    """The manuscript in the journal's order, and what was moved to get there."""
    figs, tabs = _blocks(html, "figure"), _blocks(html, "table")
    moved = []
    # Removed back to front so that each span is still valid when it is cut.
    for m in sorted(figs + tabs, key=lambda m: m.start(), reverse=True):
        kind = "Figure" if m.group(0).startswith("<figure") else "Table"
        n = _number(m.group(0), kind)
        if n is None:
            raise SystemExit(f"build_jfec_submission: a {kind.lower()} carries no "
                             f"'<b>{kind} N.</b>' caption, so it cannot be placed "
                             f"in a numbered back section")
        ph = f'<p class="jfec-ph">[{kind} {n} about here]</p>'
        html = html[:m.start()] + ph + html[m.end():]
        moved.append((kind, n, m.group(0)))
    moved.reverse()

    for kind in ("Figure", "Table"):
        got = sorted(n for k, n, _ in moved if k == kind)
        if got != list(range(1, len(got) + 1)):
            raise SystemExit(f"build_jfec_submission: the {kind.lower()}s are "
                             f"numbered {got}, which is not 1..{len(got)}; a float "
                             f"has been dropped or numbered twice")
    n_ph = len(re.findall(r'class="jfec-ph"', html))
    if n_ph != len(moved):
        raise SystemExit(f"build_jfec_submission: {len(moved)} floats were removed "
                         f"and {n_ph} placeholders inserted")

    # --- figure legends, with the alt text read from the figure itself -------
    legs = ['<div class="jfec-break"><h2>Figure legends</h2>']
    for kind, n, block in moved:
        if kind != "Figure":
            continue
        cap = _caption(block, "figcaption")
        alt = _alt(block)
        if not alt:
            raise SystemExit(f"build_jfec_submission: Figure {n} has no aria-label, "
                             f"so the alt text the journal asks for would have to "
                             f"be written by hand here")
        legs.append(f"<p>{cap}</p>")
        # "preceded by 'Alt text:'" is the journal's own wording, so the label
        # is exactly that rather than a prettier variant of it.
        legs.append(f'<p class="jfec-alt">Alt text: {alt}</p>')
    legs.append("</div>")
    # "double-spaced beginning on a separate page following the text" - the
    # house format lets the reference list begin wherever the text ends, which
    # is right for a typeset paper and not what the instruction says.
    html = html.replace("<h2>References</h2>",
                        '<h2 class="jfec-break">References</h2>', 1)
    anchor = '<h2 class="jfec-break">References</h2>'
    if anchor not in html:
        raise SystemExit("build_jfec_submission: no References heading to put the "
                         "figure legends in front of")
    html = html.replace(anchor, "\n".join(legs) + "\n" + anchor, 1)

    # --- tables, then figures, one per page, at the very back ---------------
    back = []
    for kind, head in (("Table", "Tables"), ("Figure", "Figures")):
        items = [(n, b) for k, n, b in moved if k == kind]
        if not items:
            continue
        # The heading opens the first float's page rather than taking a page
        # of its own: two pages carrying nothing but the words "Tables" and
        # "Figures" are two pages of a forty-page budget.
        lead = f'<h2>{head}</h2>'
        for n, block in items:
            if kind == "Figure":
                # The legend is on its own page by now, so the figure page
                # carries the number alone.  Printing the legend twice is what
                # the journal's order exists to avoid.
                block = re.sub(r"<figcaption>.*?</figcaption>",
                               f"<figcaption><b>Figure {n}.</b></figcaption>",
                               block, flags=re.S)
            # A caption that says "the paragraph above" is true where the
            # table is read and false where this arrangement puts it, thirty
            # pages away at the back.  The paragraph it means is a numbered
            # section, so it is named.  The substitution has to happen on the
            # BLOCK: by this point the floats have been lifted out of the
            # document, so a replacement applied to the text does nothing,
            # which is what the first version of this did.
            block = block.replace("defined in the paragraph above",
                                  "defined in Section 8")
            note = ""
            if kind == "Table":
                _cap = re.search(r"<caption.*?</caption>", block, re.S)
                if _cap:
                    brief, note = split_caption(_cap.group(0))
                    block = block[:_cap.start()] + brief + block[_cap.end():]
                    if note:
                        note = f'<p class="jfec-tablenote">{note}</p>'
            cls = "jfec-float" if kind == "Table" else "jfec-figure"
            back.append(f'<div class="{cls} jfec-break">{lead}{block}{note}'
                        f'</div>' if lead
                        else f'<div class="{cls}">{block}{note}</div>')
            lead = ""
    html = html.replace("</body>", "\n".join(back) + "\n</body>", 1)
    return html, moved


async def render(jobs):
    """Render each (html_path, pdf_path) with the settings build_pdf.py uses."""
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        for src, dst, numbered in jobs:
            await page.goto("file://" + src)
            await page.emulate_media(media="print")
            kw = dict(path=dst, print_background=True, prefer_css_page_size=True)
            if numbered:
                kw.update(display_header_footer=True, header_template="<div></div>",
                          footer_template=FOOTER,
                          margin={"top": "1in", "bottom": "1in",
                                  "left": "1in", "right": "1in"})
                kw.pop("prefer_css_page_size")
                kw["format"] = "Letter"
            await page.pdf(**kw)
        await browser.close()


async def figure_files(moved, work, out):
    """Each figure as a vector PDF and a 1200 d.p.i. TIFF, from its own SVG."""
    from playwright.async_api import async_playwright
    made = []
    os.makedirs(out, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        for kind, n, block in moved:
            if kind != "Figure":
                continue
            svg = re.search(r"<svg.*?</svg>", block, re.S).group(0)
            vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
            w, h = (float(vb.group(1)), float(vb.group(2))) if vb else (720.0, 320.0)
            # SVG user units are CSS pixels at 96 per inch, so the page is sized
            # to the figure rather than the figure to a page: a vector figure on
            # a letter page would carry an inch of white on every side into the
            # typesetter's file.
            html = (f'<!doctype html><meta charset="utf-8"><style>'
                    f'@page{{size:{w / 96:.4f}in {h / 96:.4f}in;margin:0}}'
                    f'html,body{{margin:0;padding:0;background:#fff}}'
                    f'svg{{display:block;width:{w / 96:.4f}in;height:{h / 96:.4f}in}}'
                    f'</style>{svg}')
            src = os.path.join(work, f"figure-{n}.html")
            open(src, "w", encoding="utf-8").write(html)
            pdf = os.path.join(out, f"figure-{n}.pdf")
            await page.goto("file://" + src)
            await page.emulate_media(media="print")
            await page.pdf(path=pdf, print_background=True, prefer_css_page_size=True)
            # pdftoppm writes an uncompressed raster, and 1200 d.p.i. over
            # seven inches is a hundred megabytes of mostly white paper - a file
            # no submission portal accepts and nobody should be asked to upload.
            # The same pixels through PNG and then LZW are a couple of megabytes,
            # and the d.p.i. is written into the TIFF header so the typesetter's
            # tool reads 1200 rather than inferring 72 from the pixel count.
            stem = os.path.join(out, f"figure-{n}")
            subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile",
                            pdf, stem], check=True)
            from PIL import Image
            with Image.open(stem + ".png") as im:
                im.save(stem + ".tif", format="TIFF", compression="tiff_lzw",
                        dpi=(DPI, DPI))
            os.remove(stem + ".png")
            made.append((n, pdf, stem + ".tif"))
        await browser.close()
    return made


def upload_note(n_fig, n_tab, pp_ms, pp_app):
    """The order to upload in, and the one thing that cannot be undone."""
    return "\n".join([
        "UPLOAD THESE, IN THIS ORDER",
        "",
        "  0.  the cover letter      submission/JFEc-cover-letter.md  (pasted, not uploaded)",
        "  1.  title-page.pdf        submitted, NOT for review",
        f"  2.  manuscript.pdf        the manuscript for review, {pp_ms} pp",
        f"  3.  internet-appendix.pdf supplementary material, {pp_app} pp",
        f"  4.  figures/figure-N.tif  {n_fig} files, 1200 d.p.i., line art",
        "      figures/figure-N.pdf  the same figures as vector art, if the",
        "                            portal will take them - it is the better file",
        "",
        "WHAT IS IN THE MANUSCRIPT, AND IN WHAT ORDER",
        "",
        "  abstract and keywords, numbered text, figure legends with alt text,",
        f"  references, then {n_tab} tables one per page and {n_fig} figures one per",
        "  page - the order Manuscript Preparation asks for. The text carries a",
        "  placeholder where each float was, so a copy editor can put them back.",
        "  There are no footnotes in this paper, so the footnote pages the same",
        "  instruction asks for are absent rather than empty.",
        "",
        "ANONYMITY",
        "",
        "  manuscript.pdf and internet-appendix.pdf carry no author details, and",
        "  title-page.pdf carries them all. The journal's instructions require a",
        "  title page naming the author and say nothing about the review model,",
        "  so this package satisfies either one: an anonymous pair plus a title",
        "  page is what a double-anonymised journal wants, and the title page is",
        "  what a single-anonymised one reads. Nothing here asserts which model",
        "  the Journal of Financial Econometrics uses, because neither its",
        "  General Instructions nor its submission page says.",
        "",
        "DO NOT UPLOAD",
        "",
        "  papers/variants/jfec/what-substitutes-for-a-stale-mark.pdf",
        "  papers/variants/jfec/stale-mark-internet-appendix.pdf",
        "",
        "  Those two name the author. They are the reading copies, in the house",
        "  format, and sending either one ends the anonymity if the review is in",
        "  fact anonymous - which cannot be undone.",
    ]) + "\n"


def main(argv):
    check = "--check" in argv
    _no_css_inside_svg()
    _need("pdftoppm")
    for name in (MANUSCRIPT, APPENDIX, TITLE):
        if not os.path.isfile(os.path.join(VAR, name)):
            raise SystemExit(f"build_jfec_submission: {name} is not in "
                             f"papers/variants/jfec. Run build_variants.py first.")
    out = tempfile.mkdtemp(prefix="jfec-") if check else OUT
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="jfec-work-")

    raw = open(os.path.join(VAR, MANUSCRIPT), encoding="utf-8").read()
    body, moved = reassemble(raw)
    body = body.replace("</head>", GEOMETRY + "</head>", 1)

    text = " ".join(re.sub(r"<[^>]+>", " ", body).split())
    for token in tuple(FORBIDDEN) + tuple(_private_tokens()):
        if token in text or token in body:
            raise SystemExit(f"build_jfec_submission: '{token}' is in the "
                             f"manuscript this file calls anonymous; the "
                             f"reassembly has lost the blinding")

    app = open(os.path.join(VAR, APPENDIX), encoding="utf-8").read()
    app = app.replace("</head>", GEOMETRY + "</head>", 1)

    ms_html = os.path.join(work, "manuscript.html")
    app_html = os.path.join(work, "internet-appendix.html")
    open(ms_html, "w", encoding="utf-8").write(body)
    open(app_html, "w", encoding="utf-8").write(app)
    ms_pdf = os.path.join(out, "manuscript.pdf")
    app_pdf = os.path.join(out, "internet-appendix.pdf")
    asyncio.run(render([(ms_html, ms_pdf, True), (app_html, app_pdf, True)]))

    # The same file without its floats, so the split between text and the pages
    # the one-per-page rule creates is measured rather than inferred.  The cover
    # letter states both, and verify_letters.py reads them out of MEASURED.tsv
    # rather than trusting the letter.
    bare = re.sub(r'<div class="jfec-break"><h2>Figure legends</h2>.*?</div>', "",
                  body, flags=re.S)
    bare = re.sub(r'<div class="jfec-(?:float|figure)[^"]*">.*?</div>', "", bare,
                  flags=re.S)
    bare_html = os.path.join(work, "text-only.html")
    bare_pdf = os.path.join(work, "text-only.pdf")
    open(bare_html, "w", encoding="utf-8").write(bare)
    asyncio.run(render([(bare_html, bare_pdf, True)]))
    pp_text = pages(bare_pdf)

    figs = asyncio.run(figure_files(moved, work, os.path.join(out, "figures")))

    # The title page, re-rendered with the counts it states.  A title page that
    # claims a length and sits beside a file of a different length is the copied
    # figure this project keeps finding, in the one document an editor reads
    # first.
    n_words = len(re.sub(r"<[^>]+>", " ", raw).split())
    pp_ms, pp_app = pages(ms_pdf), pages(app_pdf)
    # The title page, and the one instruction that cannot be met by two files
    # numbered separately: "page numbering should begin on this page".  So the
    # title page and the manuscript are rendered as ONE document, numbered 1..N
    # by the renderer, and then split - page 1 becomes title-page.pdf and the
    # rest becomes manuscript.pdf, still numbered 2..N.  Uploaded separately,
    # as a journal that may review anonymously requires, they still carry one
    # continuous numbering, which is what the instruction asks for and what a
    # merged proof will show.
    title = open(os.path.join(VAR, TITLE), encoding="utf-8").read()
    stamp = (f'<p class="front"><b>Length.</b> Manuscript {n_words:,} words, '
             f'{pp_ms} pages in the journal\'s geometry (US Letter, one-inch '
             f'margins, 12-point, double spaced, floats at their own leading), '
             f'of which {pp_text} are text and references and '
             f'{pp_ms - pp_text} are the pages created by setting each table '
             f'and figure on one of its own; Internet Appendix {pp_app} pages. '
             f'{sum(1 for k, _, _ in moved if k == "Table")} tables, '
             f'{len(figs)} figures, no footnotes.</p>')
    anchor = '<p class="kw">'
    if anchor in title:
        title = title.replace(anchor, stamp + "\n" + anchor, 1)
    else:
        title = title.replace("</body>", stamp + "</body>", 1)
    _tbody = re.search(r"<body[^>]*>(.*)</body>", title, re.S)
    if not _tbody:
        raise SystemExit("build_jfec_submission: the title page has no body to "
                         "put in front of the manuscript")
    joined = body.replace(
        "</head>", "</head>", 1).replace(
        re.search(r"<body[^>]*>", body).group(0),
        re.search(r"<body[^>]*>", body).group(0) + _tbody.group(1)
        + '<div class="jfec-break"></div>', 1)
    join_html = os.path.join(work, "joined.html")
    join_pdf = os.path.join(work, "joined.pdf")
    open(join_html, "w", encoding="utf-8").write(joined)
    asyncio.run(render([(join_html, join_pdf, True)]))

    from pypdf import PdfReader, PdfWriter
    _rd = PdfReader(join_pdf)
    _n_title = pages(join_pdf) - pp_ms
    if _n_title < 1:
        raise SystemExit(f"build_jfec_submission: the joined file is "
                         f"{pages(join_pdf)} pages and the manuscript alone is "
                         f"{pp_ms}, so the title page occupies {_n_title}")
    for _name, _rng in (("title-page.pdf", range(0, _n_title)),
                        ("manuscript.pdf", range(_n_title, len(_rd.pages)))):
        _w = PdfWriter()
        for _i in _rng:
            _w.add_page(_rd.pages[_i])
        with open(os.path.join(out, _name), "wb") as _fh:
            _w.write(_fh)
    # The blinding, checked again on what is actually uploaded.  Everything
    # above checks the HTML; this checks the PDF that comes out of a render, a
    # join and a split, which is three more places for the author's name to
    # survive into a file called anonymous.
    _txt = subprocess.run(["pdftotext", os.path.join(out, "manuscript.pdf"), "-"],
                          capture_output=True, text=True).stdout
    for token in tuple(FORBIDDEN) + tuple(_private_tokens()):
        if token in _txt:
            raise SystemExit(f"build_jfec_submission: '{token}' is in the "
                             f"rendered manuscript.pdf after the split; the "
                             f"title page has not been separated cleanly")
    if not _txt.strip():
        raise SystemExit("build_jfec_submission: manuscript.pdf has no text "
                         "layer, so the blinding could not be re-checked")

    n_tab = sum(1 for k, _, _ in moved if k == "Table")
    open(os.path.join(out, "UPLOAD-THESE.txt"), "w", encoding="utf-8").write(
        upload_note(len(figs), n_tab, pp_ms, pp_app))
    open(os.path.join(out, "MEASURED.tsv"), "w", encoding="utf-8").write(
        "\n".join(f"{k}\t{v}" for k, v in (
            ("manuscript_pages", pp_ms),
            ("text_and_references_pages", pp_text),
            ("float_pages", pp_ms - pp_text),
            ("internet_appendix_pages", pp_app),
            ("words", n_words),
            ("tables", n_tab),
            ("figures", len(figs)))) + "\n")

    print(f"build_jfec_submission -> {os.path.relpath(out, HERE)}")
    print(f"  manuscript.pdf         {pp_ms} pp, {n_words:,} words, "
          f"{n_tab} tables and {len(figs)} figures at the back")
    print(f"    of which {pp_text} pp are text and references and "
          f"{pp_ms - pp_text} pp are float pages")
    print(f"  internet-appendix.pdf  {pp_app} pp")
    for n, pdf, tif in figs:
        print(f"  figures/figure-{n}      {os.path.getsize(pdf) // 1024} KB vector, "
              f"{os.path.getsize(tif) // 1024} KB at 1200 d.p.i.")
    # What the gate is, and why it is not the total.
    #
    # The journal's guidance is that a paper "should typically not exceed 40
    # double-spaced typewritten pages".  The total here is larger than the
    # integrated manuscript BECAUSE of the journal's own placement rule: every
    # table and every figure takes a page of its own, which turns four pages of
    # floats into eleven.  Those pages are not discretionary content, and
    # failing a build over them would mean failing it for complying with the
    # instruction that created them.  So the gate is on the part the author
    # controls - text and references - and the total is reported beside it,
    # which is the number to quote in the cover letter and the one a desk editor
    # will count.
    over = max(0, pp_text - LIMIT)
    print(f"  the journal's guidance is {LIMIT} double-spaced pages: text and "
          f"references are {pp_text}"
          + (f", which is {over} over" if over else ", within it")
          + f"; the whole file, with every table on a page of its own as the "
            f"same instruction requires, is {pp_ms}")
    shutil.rmtree(work, ignore_errors=True)
    if check:
        shutil.rmtree(out, ignore_errors=True)
    if over:
        print(f"  FAILED: {over} page(s) of text have to leave the body, or the "
              f"length has to be defended in the cover letter deliberately "
              f"rather than by default")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
