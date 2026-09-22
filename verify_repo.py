"""
verify_repo.py - what ships, and whether the README describes it honestly.

    python verify_repo.py
    python verify_repo.py --manifest    one line per file that must be public

WHY THIS FILE EXISTS
--------------------
The README is the first thing a reader sees and the last thing anyone updates.
When it was written the suite had forty-nine scripts; by the time anyone looked
again it had sixty-three, the registry had gained a fifth gate, and four
top-level scripts existed that the README did not mention at all.  None of that
is visible while reading, because a stale count reads exactly like a current
one.  It is this project's one recurring fault - a number copied rather than
referenced - committed against the document that introduces the project.

What is checked:

    * every count of labs or scripts, in digits and in words
    * the number of registry gates, against build_registry.py's own gate list
    * the number of rows in the lab table, against labs/
    * that every top-level .py file is named somewhere in the README
    * that every file the README names in backticks actually exists

    * that nothing ships which is not part of reproducing the result

The last of those is why this file is not called verify_readme.py.  A repository
whose job is reproduction earns its size: sixty-three labs and the sixty-three
stored outputs they are diffed against are the reproduction, not clutter.  What
does not belong is a SECOND COPY of something - and one was there: a hidden
papers/source/.pre-jfec/ holding a three-day-old snapshot of all three
manuscripts, referenced by nothing, 817 diff lines away from the live source.
That is this project's one recurring fault given a directory of its own.  A
snapshot taken "just in case" before a risky edit is exactly how a reader ends
up reading the wrong manuscript, so the check below refuses hidden directories
and duplicate manuscript sources rather than trusting anyone to remember.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(HERE, "README.md")

WORDS = {"forty-nine": 49, "fifty-four": 54, "fifty-five": 55, "fifty-nine": 59,
         "sixty": 60, "sixty-one": 61, "sixty-two": 62, "sixty-three": 63,
         "sixty-four": 64, "sixty-five": 65,
         "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
SPELL = {v: k for k, v in WORDS.items()}


def _gitignored():
    """The .gitignore entries, as bare names - what is absent on purpose."""
    return [l.strip().rstrip("/") for l in
            open(os.path.join(HERE, ".gitignore"), encoding="utf-8")
            if l.strip() and not l.startswith("#")]


def main(argv):
    txt = open(README, encoding="utf-8").read()
    labs = sorted(f for f in os.listdir(os.path.join(HERE, "labs"))
                  if re.fullmatch(r"lab\d+[a-z]?_.*\.py", f))
    n_labs = len(labs)
    # the gate count comes from build_registry.py's own numbered list, so the
    # README cannot disagree with the thing it is describing
    reg = open(os.path.join(HERE, "build_registry.py"), encoding="utf-8").read()
    n_gates = len(re.findall(r"^\d+\. [A-Z][A-Z ]", reg, re.M))
    fails = []

    # 1. counts of labs and scripts, in digits and spelled out.
    #    Two different totals live in this README and they are NOT the same
    #    number: sixty-three labs EXIST, and forty-nine of them are CITED by the
    #    main paper.  verify_paper.py pins the cited count from its own set of
    #    quoted figures, so "one of forty-nine scripts" is correct and rewriting
    #    it to sixty-three - which is what happened while correcting the genuinely
    #    stale counts around it - broke a check two files away.  A count is
    #    therefore read against the total its own sentence is about.
    cited = re.search(r"one of ([a-z-]+) scripts", txt)
    n_cited = WORDS.get(cited.group(1)) if cited else None
    for m in re.finditer(r"\b(\d{2})\s+(?:labs|scripts)\b", txt):
        if int(m.group(1)) != n_labs:
            fails.append(f"README says {m.group(1)} labs/scripts; there are {n_labs}")
    for m in re.finditer(r"\b([a-z]+(?:-[a-z]+)?)\s+(?:labs|scripts)\b", txt):
        w = m.group(1)
        if n_cited is not None and WORDS.get(w) == n_cited and \
                re.search(r"one of " + re.escape(w), txt):
            continue                      # the cited count, not the total
        if w in WORDS and WORDS[w] != n_labs:
            # "Four of the sixty-three labs" is a subset, not a total: only the
            # word IMMEDIATELY before labs/scripts is read as the total, and a
            # subset phrase names its total right after it, which this catches.
            if not re.search(re.escape(w) + r"\s+of\s+the", txt):
                fails.append(f"README says '{w} labs/scripts'; there are "
                             f"{n_labs} ({SPELL.get(n_labs, n_labs)})")

    # 2. the registry's gate count
    for m in re.finditer(r"\b(\w+)\s+(?:build\s+)?gates\b", txt):
        w = m.group(1).lower()
        got = WORDS.get(w, int(w) if w.isdigit() else None)
        if got is not None and got != n_gates:
            fails.append(f"README says {w} gates; build_registry.py defines "
                         f"{n_gates}")

    # 3. the lab table lists every lab, and only real ones
    rows = set(re.findall(r"^\| `(lab[^`]+)`", txt, re.M))
    for missing in sorted(set(labs) - rows):
        fails.append(f"the lab table has no row for {missing}")
    for ghost in sorted(rows - set(labs)):
        fails.append(f"the lab table has a row for {ghost}, which is not in labs/")

    # 4. every top-level script is mentioned
    for f in sorted(os.listdir(HERE)):
        # No self-exemption.  The earlier version skipped itself, which is
        # how it passed while the README did not mention it at all - a checker
        # that excuses itself from its own rule is a checker with a blind spot
        # exactly where it has most authority.
        if f.endswith(".py") and f not in txt:
            fails.append(f"{f} is in the repository and not named in the README")

    # 5. every backticked filename the README names exists - or is a vendor file
    #    the repository deliberately does not ship, or a placeholder.  The
    #    manifest is what distinguishes the two: a name listed there is an input
    #    a reader downloads themselves, so its absence is the licence position
    #    and not a broken pointer.
    have = set()
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in ("output", "__pycache__", ".git")]
        for f in files:
            have.add(f)
            have.add(os.path.relpath(os.path.join(root, f), HERE).replace("\\", "/"))
    man = open(os.path.join(HERE, "data", "MANIFEST.tsv"), encoding="utf-8").read()
    for name in sorted(set(re.findall(r"`([A-Za-z0-9_./<>-]+\.(?:py|tsv|md|bat|txt|"
                                      r"csv|html|pdf))`", txt))):
        base = os.path.basename(name)
        placeholder = bool(re.search(r"XYZ|<[^>]+>", name))
        # A name the .gitignore excludes is absent ON PURPOSE - contact.txt is
        # the case, and the README explains why it is not here.  Missing this
        # made a clean clone fail its own make_all on the very first step,
        # which a clone test caught and reading never would have: the author's
        # working copy has the file, so the check passed everywhere it was run.
        withheld = any(name == g or base == g or name.startswith(g + "/")
                       for g in _gitignored())
        if name in have or base in have or base in man or placeholder or withheld:
            continue
        fails.append(f"the README names `{name}`, which does not exist and is "
                     f"not an input listed in the manifest")

    # 6. every italic cross-reference to one of the README's own sections
    #    resolves.  This one earns its place: the opening paragraph was once
    #    edited down to the single word "below.", pointing at nothing, and it
    #    sat in the first five lines of the repository's front page.
    #    The first version of this excluded newlines from the reference text,
    #    so it never matched: the one cross-reference in the file wraps across
    #    two lines, and the check passed on every input including a deliberately
    #    broken one.  A check that cannot fire is worse than none, because it
    #    reads as coverage.  Newlines are now allowed and whitespace collapsed.
    heads = {h.strip().lower() for h in re.findall(r"^#+\s+(.+)$", txt, re.M)}
    for ref in re.findall(r"\*([A-Z][^*]{12,120}?)\*\s+below", txt, re.S):
        if " ".join(ref.split()).lower() not in heads:
            fails.append(f"the README points at a section "
                         f"'{' '.join(ref.split())}' below, "
                         f"which is not one of its headings")

    # 7. nothing ships that is not part of reproducing the result.
    #    Three rules, each one a thing that was actually found here or is the
    #    obvious next version of it.
    ign = _gitignored()
    shipped = []
    for root, dirs, files in os.walk(HERE):
        rel_root = os.path.relpath(root, HERE).replace("\\", "/")
        dirs[:] = [d for d in dirs if not any(
            d == p or (rel_root + "/" + d).lstrip("./") == p for p in ign)]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), HERE).replace("\\", "/")
            if not any(rel == p or rel.startswith(p + "/") or f == p or
                       (p.startswith("*") and f.endswith(p[1:])) for p in ign):
                shipped.append(rel)

    #    (a) no hidden directory ships.  A dot-prefixed folder is invisible in a
    #        file listing and on GitHub's web view, which is what let a snapshot
    #        of the manuscripts sit in papers/source/ unnoticed.
    for rel in shipped:
        parts = rel.split("/")[:-1]
        if any(p.startswith(".") for p in parts):
            fails.append(f"{rel} ships from a hidden directory; a folder nobody "
                         f"sees is where stale copies survive")

    #    (b) no second copy of a manuscript source.  The published source lives
    #        in papers/source/ and nowhere else; a duplicate under any other
    #        path is a copy that will drift.
    stems = ("what-substitutes-for-a-stale-mark", "stale-mark-internet-appendix",
             "threshold-rule-at-its-ceiling")
    for rel in shipped:
        if rel.endswith(".html") and any(s in rel for s in stems) \
                and not rel.startswith("papers/source/"):
            fails.append(f"{rel} is a second copy of a manuscript source; the "
                         f"one that counts is in papers/source/")

    #    (c) every lab ships with the stored output it is diffed against, and
    #        every stored output with its lab.  An orphan on either side means
    #        run_all.py --check is quietly not checking something.
    stored = {f[:-4] for f in os.listdir(os.path.join(HERE, "expected_output"))
              if f.endswith(".txt")}
    for lab in labs:
        if lab[:-3] not in stored:
            fails.append(f"{lab} has no stored output, so --check cannot diff it")
    for s in sorted(stored - {l[:-3] for l in labs}):
        fails.append(f"expected_output/{s}.txt has no lab, so it is diffed "
                     f"against nothing")

    # 8. the GitHub "About" line.  It lives on the website, not in the
    #    repository, so no checker here can read it - and that is exactly why
    #    it went stale twice: first naming a paper title that no longer
    #    existed, then claiming "46 scripts" when there are 63, 57
    #    non-exploratory and 49 cited by the main paper, so the figure matched
    #    nothing at all.  It is also the single most-read sentence in the whole
    #    project, since it sits beside the URL the papers print.  So the
    #    canonical text lives HERE, in repo-about.txt, its number is checked
    #    against the labs, and publishing it is a copy-paste rather than a
    #    retype.  GitHub truncates the description at 350 characters.
    _about = os.path.join(HERE, "repo-about.txt")
    if not os.path.isfile(_about):
        fails.append("repo-about.txt is missing; the GitHub description has no "
                     "checked source and will drift again")
    else:
        _a = open(_about, encoding="utf-8").read().strip()
        if len(_a) > 350:
            fails.append(f"repo-about.txt is {len(_a)} characters; GitHub "
                         f"truncates the description at 350")
        _n = re.search(r"\b(\d+) scripts\b", _a)
        if not _n:
            fails.append("repo-about.txt does not state a script count, so "
                         "nothing ties it to the labs")
        elif int(_n.group(1)) != n_labs:
            fails.append(f"repo-about.txt says {_n.group(1)} scripts; there "
                         f"are {n_labs}")

    # 9. the three documents must state the SAME date, and it must be a date.
    #    Each legitimately carries its own date line, so there is no single
    #    source to point them at - which makes them three copies that have to
    #    be checked rather than trusted.  A fourth copy was sitting in
    #    build_variants.py as a literal, on the title page that introduces the
    #    manuscript to an editor; it now reads the manuscript's own line.  The
    #    date moved the day after it was written, which is how often this needs
    #    to be right.
    _dates = {}
    for _f, _pat in (
            ("what-substitutes-for-a-stale-mark.html",
             r'<p class="date">([^<]+)</p>'),
            ("threshold-rule-at-its-ceiling.html",
             r'<p class="date">([^<]+)</p>'),
            ("stale-mark-internet-appendix.html",
             r'<p class="t" style="text-align:center">(\d[^<]*\d)</p>')):
        _p = os.path.join(HERE, "papers", "source", _f)
        if os.path.isfile(_p):
            _m = re.search(_pat, open(_p, encoding="utf-8").read())
            _dates[_f] = _m.group(1).strip() if _m else None
    for _f, _d in _dates.items():
        if not _d:
            fails.append(f"{_f} states no date on its title block")
        elif not re.fullmatch(r"\d{1,2} [A-Z][a-z]+ \d{4}", _d):
            fails.append(f"{_f} dates itself '{_d}', which is not a date")
    _set = {d for d in _dates.values() if d}
    if len(_set) > 1:
        fails.append("the three documents disagree about their date: "
                     + "; ".join(f"{k} says {v}" for k, v in _dates.items()))

    # --manifest prints exactly what must be public, one path per line, so
    # checking a published repository against this one is a comparison rather
    # than an eyeball.  Two files went missing from the first upload of this
    # project - check_on_windows.bat and results_registry.tsv - and nothing
    # noticed, because a repository that is missing a file looks exactly like
    # a repository that never had it.
    if "--manifest" in argv:
        for rel in sorted(shipped):
            print(rel)
        return 0

    print(f"verify_repo: {len(shipped)} files ship, {n_labs} labs, "
          f"{n_gates} registry gates, {len(rows)} table rows")
    for f in fails:
        print("  FAIL  " + f)
    if not fails:
        print("  the README agrees with the repository it describes")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
