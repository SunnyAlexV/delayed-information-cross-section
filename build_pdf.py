"""
build_pdf.py - render the papers from their HTML source.

    python build_pdf.py            render both papers into papers/
    python build_pdf.py --check    render to a temporary file and report the page
                                   count without touching papers/

Why this file exists.  Both papers claim that every number in them is produced by
a script and checked by another one, and until now the one step that was not
scripted was turning the source into the PDF that people actually read.  That
step was done by hand, which meant the render settings lived in somebody's shell
history: a later render with different settings produces a different document
from the same source, and nothing would have caught it.

The settings matter more than they look.  Rendering this source with
wkhtmltopdf gives 23 pages instead of 49 and silently drops the print
stylesheet; rendering it with Chromium's --print-to-pdf gives the right page
count but more than twice the file size.  The combination below - Chromium
through Playwright, print media emulated, backgrounds on, and the page size
taken from the CSS rather than from the default paper - is the one that
reproduces the shipped documents.

Requires playwright and its Chromium.  If the browser is missing, the error
says so rather than rendering something subtly different.
"""

import asyncio
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "papers", "source")
OUT = os.path.join(HERE, "papers")

PAPERS = ["what-substitutes-for-a-stale-mark", "stale-mark-internet-appendix",
          "threshold-rule-at-its-ceiling"]


def pages(path):
    """Page count, read out of the PDF itself rather than reported by the tool."""
    return len(re.findall(rb"/Type\s*/Page[^s]", open(path, "rb").read()))


async def render(names, out_dir, src_dir=None):
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise SystemExit(
            "playwright is not installed, so the papers cannot be rendered.\n"
            "  pip install playwright --break-system-packages\n"
            "The HTML source in papers/source is complete and readable without it.")
    done = []
    async with async_playwright() as pw:
        try:
            browser = await pw.chromium.launch()
        except Exception as exc:
            raise SystemExit(
                f"Playwright could not start Chromium: {exc}\n"
                "This environment normally provides it at PLAYWRIGHT_BROWSERS_PATH; "
                "do not substitute another renderer, because the page geometry "
                "differs and the shipped documents would not be reproduced.")
        page = await browser.new_page()
        for name in names:
            src = os.path.join(src_dir or SRC, name + ".html")
            if not os.path.isfile(src):
                raise SystemExit(f"missing source: {src}")
            dst = os.path.join(out_dir, name + ".pdf")
            await page.goto("file://" + src)
            await page.emulate_media(media="print")
            await page.pdf(path=dst, print_background=True, prefer_css_page_size=True)
            done.append((name, dst, pages(dst), os.path.getsize(dst)))
        await browser.close()
    return done


def main(argv):
    check = "--check" in argv
    # --src and --out let the journal variants render through THIS file rather
    # than a second copy of the settings.  The whole reason this script exists
    # is that render settings kept in a shell history produce a different
    # document from the same source; a variant renderer would reintroduce that
    # exact fault three times over.
    # Both are made absolute here.  A relative --src reached page.goto() as
    # "file://papers/..." and Chromium rejected it as an invalid URL, which is
    # the right failure but an obscure one to read.
    def _flag(name):
        v = next((a.split("=", 1)[1] for a in argv[1:]
                  if a.startswith(name + "=")), None)
        return os.path.abspath(v) if v else None
    src_dir, out_arg = _flag("--src"), _flag("--out")
    names = [a for a in argv[1:] if not a.startswith("--")] or PAPERS
    out_dir = out_arg or (os.path.join(HERE, "output") if check else OUT)
    os.makedirs(out_dir, exist_ok=True)
    for name, dst, n, size in asyncio.run(render(names, out_dir, src_dir)):
        where = (os.path.relpath(out_dir, HERE) + "/"
                 if out_arg else ("output/ (check only)" if check else "papers/"))
        print(f"  {name}.pdf  {n} pages, {size/1024:.0f} KB  -> {where}")
    if check:
        print("\n  --check wrote into output/ and left papers/ alone.")


if __name__ == "__main__":
    main(sys.argv)
