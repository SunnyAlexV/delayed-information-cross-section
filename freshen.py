"""freshen.py - rebuild everything, then extract what the EDITOR will read.

    python freshen.py            rebuild, re-verify, extract
    python freshen.py --extract  extract only (when a rebuild just ran)

WHY THIS FILE EXISTS
--------------------
There are three copies of this paper at any moment: papers/source/, which is
what gets edited; papers/variants/<key>/, which is generated from it; and
papers/submission/jfec/, which is what a journal actually receives.  They
disagree until build_variants.py and build_jfec_submission.py have run, and
an answer read off the wrong one is wrong in a way that looks right.

That has now caused two errors in one day.  The specification band was checked
against Table 1 in papers/source/ and the table was re-read from a copy that
predated the edit; and the band's own rows turned out to be quoted on two
different dating conventions, which a check of the endpoints alone could not
see.

So this file removes the choice.  It rebuilds in the documented order, fails
loudly if any gate fails, and then writes ONE text file holding the current
plain text of every document the editor receives, with a header saying when it
was made and what the verifiers said at that moment.  Any question about what
the paper says is answered from that file and from nothing else.
"""

import datetime
import html as _html
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "FRESH")

# What the editor receives, in the order they meet it.  The cover letter is
# included because half the contradictions found so far were between the letter
# and the documents it describes, and nothing read them together.
DOCS = [
    ("COVER LETTER (pasted into the portal)",
     os.path.join(HERE, "submission", "JFEc-cover-letter.md")),
    ("TITLE PAGE (uploaded, not for review)",
     os.path.join(HERE, "papers", "submission", "jfec", "title-page.pdf")),
    ("MANUSCRIPT (for review)",
     os.path.join(HERE, "papers", "submission", "jfec", "manuscript.pdf")),
    ("INTERNET APPENDIX (for review)",
     os.path.join(HERE, "papers", "submission", "jfec", "internet-appendix.pdf")),
    ("UPLOAD MANIFEST",
     os.path.join(HERE, "papers", "submission", "jfec", "UPLOAD-THESE.txt")),
]


def run(name, args, why):
    print(f"  {name:28} {why}")
    r = subprocess.run([sys.executable, os.path.join(HERE, name)] + args,
                       capture_output=True, text=True, cwd=HERE)
    tail = (r.stdout + r.stderr).strip().splitlines()
    if r.returncode != 0:
        print("\n".join("      " + l for l in tail[-14:]))
        raise SystemExit(f"freshen: {name} failed; nothing downstream of it is "
                         f"current, so the extract would describe a state that "
                         f"does not exist")
    return tail


def text_of(path):
    if not os.path.isfile(path):
        return f"[MISSING: {path}]"
    if path.endswith(".pdf"):
        r = subprocess.run(["pdftotext", "-layout", path, "-"],
                           capture_output=True, text=True)
        return r.stdout
    raw = open(path, encoding="utf-8").read()
    if path.endswith(".html"):
        raw = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw)
        raw = _html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return raw


def main(argv):
    os.makedirs(OUT, exist_ok=True)
    verdicts = []
    if "--extract" not in argv:
        print("rebuilding, in the documented order:")
        for name, args, why in (
                ("build_variants.py", [], "regenerate the three variants"),
                ("build_jfec_submission.py", [],
                 "assemble the JFEc package in its own order"),
                ("verify_paper.py",
                 [os.path.join(HERE, "papers", "variants", "jfec",
                               "what-substitutes-for-a-stale-mark.html"),
                  os.path.join(HERE, "papers", "source",
                               "why-a-simple-threshold-rule-is-hard-to-beat.html")],
                 "every figure against its lab output"),
                ("verify_letters.py", [], "the letters against the package"),
                ("verify_repo.py", [], "the repository against the README")):
            tail = run(name, args, why)
            verdicts.append((name, tail[-2:] if len(tail) > 1 else tail))

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        "=" * 78,
        f"FRESH EXTRACT  {stamp}",
        "",
        "The current text of every document the editor receives, taken AFTER a",
        "full rebuild.  Answer questions about what the paper says from this",
        "file, not from papers/source/ and not from memory of an earlier read.",
        "=" * 78,
        "",
        "VERIFIER STATE AT EXTRACT TIME",
    ]
    for name, tail in verdicts:
        for line in tail:
            parts.append(f"  {name:26} {line.strip()}")
    parts.append("")

    for label, path in DOCS:
        parts += ["", "=" * 78, f"### {label}",
                  f"### {os.path.relpath(path, HERE)}", "=" * 78, "",
                  text_of(path)]

    dst = os.path.join(OUT, "EDITOR-VIEW.txt")
    open(dst, "w", encoding="utf-8").write("\n".join(parts))
    n = len(open(dst, encoding="utf-8").read().split())
    print(f"\n  wrote {os.path.relpath(dst, HERE)}  ({n:,} words, {stamp})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
