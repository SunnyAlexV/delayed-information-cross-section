"""
measure_pages.py - page count in the geometry a JOURNAL counts in.

    python measure_pages.py                    the three journal variants
    python measure_pages.py papers/source      any directory holding the sources

Why this file exists.  A page limit in a submission guideline is not a count of
the pages in the PDF the author renders.  It is a count of pages in a specific
typewritten geometry - US Letter, one-inch margins, 12-point serif, double
spaced - which is roughly what a manuscript occupied when it arrived as paper.
This paper's own stylesheet is a house format: A4, 10.5pt, single spaced,
justified, with tight tables.  The two counts differ by well over half again,
and the difference is not an error in either: the same manuscript is 27 pages in
one geometry and 43 in the other.

That gap is worth a script rather than an argument.  Asked whether the paper is
under a forty-page limit, the honest answer depends entirely on which geometry
the question means, and the only way to stop guessing is to render both and
report both.  The override below is applied on top of the source rather than
edited into it, so the shipped documents keep their own typography and the
measurement never alters what it measures.

Two columns are reported, because the guideline itself is ambiguous about
floats.  The first leaves tables and figures at their own leading, which is what
every journal's typesetting actually does; the second doubles every line in the
file, which is what a desk editor gets by selecting all and pressing the
double-space button.  Neither is the "true" count - the count is between them,
and a single number here would be a preference dressed as a measurement.
"""

import asyncio
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = "what-substitutes-for-a-stale-mark"
SUPP = "stale-mark-internet-appendix"

# US Letter, one-inch margins, 12pt serif, double spaced.  @page wins over the
# source's own @page because it is appended last; !important is needed on the
# body rules because the source sets them on element selectors of equal weight.
OVERRIDE = """
<style id="journal-geometry">
@page { size: 8.5in 11in; margin: 1in; }
body, p, li, td, th, div.abstract, .front {
  font-family: "Times New Roman", Times, serif !important;
  font-size: 12pt !important;
  text-align: left !important;
}
body, p, li { line-height: 2.0 !important; }
/* Floats keep their own leading: no journal double-spaces a table body. */
table, table *, figure, figure *, caption, .cap, .note, .fn {
  line-height: 1.15 !important; font-size: 10pt !important;
}
h1, h2, h3 { line-height: 1.4 !important; }
</style>
"""


def pages(path):
    """Read the page count out of the PDF, not out of the renderer's report."""
    return len(re.findall(rb"/Type\s*/Page[^s]", open(path, "rb").read()))


# The literal reading of the same guideline: EVERYTHING double spaced, floats
# included.  No journal desk actually does this, but a desk editor running a
# word processor's "double space" command on the whole file does, so the two
# renders bracket the number rather than guessing at it.  Reporting one of them
# alone is how a page count becomes an argument.
STRICT = OVERRIDE.replace(
    'table, table *, figure, figure *, caption, .cap, .note, .fn {\n'
    '  line-height: 1.15 !important; font-size: 10pt !important;\n}',
    'table, table *, figure, figure *, caption, .cap, .note, .fn {\n'
    '  line-height: 2.0 !important; font-size: 12pt !important;\n}')
assert STRICT != OVERRIDE, "the strict override stopped matching OVERRIDE"


async def measure(items, tmp, css=None):
    from playwright.async_api import async_playwright
    css = css or OVERRIDE
    out = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        for label, src in items:
            html = open(src, encoding="utf-8").read()
            # appended last so it overrides, and inside <head> so it applies to
            # the print stylesheet rather than arriving after first layout
            shim = html.replace("</head>", css + "</head>", 1)
            if shim == html:                       # no <head>: prepend instead
                shim = css + html
            tag = "strict" if css is STRICT else "floats"
            work = os.path.join(tmp, label.replace("/", "_") + "-" + tag + ".html")
            open(work, "w", encoding="utf-8").write(shim)
            pdf = work[:-5] + ".pdf"
            await page.goto("file://" + work)
            await page.emulate_media(media="print")
            await page.pdf(path=pdf, print_background=True,
                           prefer_css_page_size=True)
            out.append((label, pages(pdf)))
        await browser.close()
    return out


def main(argv):
    dirs = [a for a in argv[1:] if not a.startswith("-")]
    if not dirs:
        base = os.path.join(HERE, "papers", "variants")
        dirs = [os.path.join(base, d) for d in sorted(os.listdir(base))
                if os.path.isdir(os.path.join(base, d))]
    items = []
    for d in dirs:
        for name in (PAPER, SUPP):
            p = os.path.join(os.path.abspath(d), name + ".html")
            if os.path.isfile(p):
                items.append((os.path.basename(os.path.abspath(d)) + "/" + name, p))
    tmp = os.path.join(HERE, "output", "geometry")
    os.makedirs(tmp, exist_ok=True)
    floors = dict(asyncio.run(measure(items, tmp)))
    strict = dict(asyncio.run(measure(items, tmp, STRICT)))
    print(f"{'document':42}{'floats single':>15}{'all double':>13}")
    for label, _ in items:
        print(f"  {label:40}{floors[label]:>13} pp{strict[label]:>11} pp")
    print("\n  The two columns bracket the count: floats at their own leading,\n"
          "  and the literal reading that doubles every line in the file.")


if __name__ == "__main__":
    main(sys.argv)
