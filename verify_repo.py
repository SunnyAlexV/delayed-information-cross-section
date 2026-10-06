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

import ast
import csv
import os
import re
import subprocess
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
             "why-a-simple-threshold-rule-is-hard-to-beat")
    for rel in shipped:
        if rel.endswith(".html") and any(s in rel for s in stems) \
                and not rel.startswith("papers/source/"):
            fails.append(f"{rel} is a second copy of a manuscript source; the "
                         f"one that counts is in papers/source/")

    #    (b2) no second copy of a LAB ships.  Rule (b) above covers the
    #        manuscripts and rule (c) below checks labs/ against
    #        expected_output/, so a lab_NN_*.py sitting under some OTHER
    #        directory was invisible to every rule here - and one was: an
    #        identical copy of a wired lab sat in a working folder, free to
    #        drift from its live twin, while a reader counting scripts would
    #        have found more of them than the reproducibility claim covers.
    #        A lab counts only where run_all.py can see it.
    for rel in shipped:
        base = os.path.basename(rel)
        if re.fullmatch(r"lab\d+[a-z]?_.*\.py", base) and \
                not rel.startswith("labs/"):
            fails.append(f"{rel} is a lab outside labs/, so run_all.py never "
                         f"runs it and nothing diffs it; either wire it into "
                         f"labs/ with its output stored or keep it out of the "
                         f"repository")

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

    #    (c2) every lab is NAMED IN run_all.py's own table.  Rules (b2) and (c)
    #        read two surfaces - where a lab file sits, and whether a stored
    #        output exists beside it - and a lab can satisfy both while being
    #        invisible to the command that reproduces the paper.  One was:
    #        lab67 sat in labs/ with its output in expected_output/, so the
    #        count of scripts was right, the README table had its row, nothing
    #        was orphaned, and `run_all.py --check` never ran it, because the
    #        one surface no rule read was the LABS list inside run_all.py.
    #        The paper meanwhile quoted its figures. This is the same failure
    #        class as a check pinned to the wrong sentence: the check was real,
    #        it just read a surface next to the one that mattered.
    _ra = open(os.path.join(HERE, "run_all.py"), encoding="utf-8").read()
    _tabled = set(re.findall(r'\(\s*"(lab\d+[a-z]?_[^"]+\.py)"', _ra))
    for lab in labs:
        if lab not in _tabled:
            fails.append(f"{lab} is not in run_all.py's LABS table, so "
                         f"`run_all.py --check` never runs or diffs it, "
                         f"whatever the paper quotes from it")
    for lab in sorted(_tabled - set(labs)):
        fails.append(f"run_all.py's LABS table names {lab}, which is not in "
                     f"labs/, so a full run stops on it")

    #    (c3) every section and table the labs and this README cite EXISTS in
    #        one of the three documents.  verify_paper.py has a cross-reference
    #        guard, and it reads the two papers only; nothing read the labs,
    #        and 117 references had accumulated to sections and tables from two
    #        earlier numberings of the manuscript - "Section 9.2", "Table 17" -
    #        several of them inside print() calls, so they reached every reader
    #        of the replication package.  A dangling reference is cheap to
    #        detect and impossible to notice by reading, which is the whole
    #        argument for checking it here.
    #
    #        This checks only that the target EXISTS.  Whether it holds the
    #        cited subject is verify_paper.py's job, and it does that for the
    #        papers' own references; a lab citing a real section that does not
    #        discuss the matter would pass here and is the residual risk.
    _docs = {}
    for _f, _k in (("what-substitutes-for-a-stale-mark.html", "paper"),
                   ("stale-mark-internet-appendix.html", "supp"),
                   ("why-a-simple-threshold-rule-is-hard-to-beat.html", "note")):
        _p = os.path.join(HERE, "papers", "source", _f)
        _docs[_k] = open(_p, encoding="utf-8").read() if os.path.isfile(_p) else ""
    # subsection headings carry NO period after the number ("<h3>4.1 The
    # volatility estimator"), so the boundary is a period OR whitespace; an
    # earlier version of this pattern required the period, found no subsections
    # at all, and would have reported every correct "Section 4.6" as dangling
    _SEC = r"<h[23][^>]*>(%s\d+(?:\.\d+)?)[\.\s]"
    _ok_sec = ({m.group(1) for m in re.finditer(_SEC % "", _docs["paper"])}
               | {m.group(1) for m in re.finditer(_SEC % "", _docs["note"])})
    _ok_ssec = {m.group(1) for m in re.finditer(_SEC % "S", _docs["supp"])}
    _TAB = r"<b>Table (%s\d+(?:\.\d+)?)\.</b>"
    _ok_tab = ({m.group(1) for m in re.finditer(_TAB % "", _docs["paper"])}
               | {m.group(1) for m in re.finditer(_TAB % "", _docs["note"])})
    _ok_stab = {m.group(1) for m in re.finditer(_TAB % "S", _docs["supp"])}
    if len(_ok_sec) < 15 or len(_ok_ssec) < 30 or len(_ok_stab) < 20:
        fails.append(f"could not index the documents' own sections and tables "
                     f"for the citation check ({len(_ok_sec)} paper/note "
                     f"sections, {len(_ok_ssec)} appendix, {len(_ok_stab)} "
                     f"appendix tables), so the check below cannot fail")
    else:
        _dangle = []
        for _rel in sorted([os.path.join("labs", f) for f in labs]
                           + ["README.md"]):
            _src = open(os.path.join(HERE, _rel), encoding="utf-8").read()
            for _kind, _pat, _plain, _supp in (
                    ("Section", r"\bSection (S?\d+(?:\.\d+)?)\b",
                     _ok_sec, _ok_ssec),
                    ("Table", r"\bTable (S?\d+(?:\.\d+)?)\b",
                     _ok_tab, _ok_stab)):
                for _m in re.finditer(_pat, _src):
                    _r = _m.group(1)
                    if _r in (_supp if _r.startswith("S") else _plain):
                        continue
                    # one deliberate historical mention: lab48's docstring
                    # quotes the wrong citation it is recording as an error,
                    # and is exempt BY ITS WORDING, not by file name
                    _ctx = _src[max(0, _m.start() - 170):_m.start() + 40]
                    if "cited as" in _ctx and "exists in" in _src[
                            _m.start():_m.start() + 120]:
                        continue
                    _dangle.append(f"{_rel}: {_kind} {_r}")
        if _dangle:
            _by = {}
            for _d in _dangle:
                _by.setdefault(_d.split(": ")[1], []).append(
                    _d.split(":")[0])
            fails.append(
                f"{len(_dangle)} citation(s) in labs/ and README.md point at a "
                f"section or table that exists in none of the three documents: "
                + "; ".join(f"{_k} ({len(set(_v))} file(s))"
                            for _k, _v in sorted(_by.items())))

    # 7a2. EVERY moving-block bootstrap draws its starts on an INCLUSIVE upper
    #      end and covers the sample.  Two sites had neither: lab67 part C and
    #      joint_null's demonstration both used `n // block` blocks, which
    #      makes a replication shorter than the sample the observed statistic
    #      was computed on, and `integers(0, n - block)`, which is exclusive
    #      and so never draws the last observation.  Both biased lab68's null
    #      count downward, which is the direction that flattered the paper.
    #
    #      A sweep found exactly those two among thirty-four bootstraps; every
    #      other lab already had it right.  That is the argument for checking
    #      it rather than trusting it: the two wrong ones were the two most
    #      recently written, and the next lab written will be new too.
    _BAD_HI = re.compile(r"\.(?:integers|randint)\(\s*0\s*,\s*"
                         r"(?:max\(\s*1\s*,\s*)?"
                         r"([A-Za-z_]\w*)\s*-\s*([A-Za-z_]\w*)\s*(?:\)\s*)?,")
    _BAD_NB = re.compile(r"^\s*(?:nb|nblk|nbk)\s*=\s*(?:max\(\s*1\s*,\s*)?"
                         r"([A-Za-z_]\w*)\s*//\s*([A-Za-z_]\w*)")
    for _nm in sorted(f for f in os.listdir(os.path.join(HERE, "labs"))
                      if f.endswith(".py")):
        _src = open(os.path.join(HERE, "labs", _nm), encoding="utf-8").read()
        for _i, _ln in enumerate(_src.splitlines(), 1):
            if _ln.lstrip().startswith("#"):
                continue
            _m = _BAD_HI.search(_ln)
            if _m and _m.group(2).lower() in ("block", "blk", "l", "b"):
                fails.append(f"labs/{_nm}:{_i} draws block starts on "
                             f"integers(0, {_m.group(1)} - {_m.group(2)}), whose "
                             f"upper end is EXCLUSIVE, so the last observation "
                             f"is never drawn; the moving-block range is "
                             f"[0, n - block] and needs + 1")
            _m2 = _BAD_NB.search(_ln)
            if _m2:
                fails.append(f"labs/{_nm}:{_i} sets the block count by floor "
                             f"division, {_m2.group(1)} // {_m2.group(2)}, so "
                             f"every replication is shorter than the sample the "
                             f"observed statistic was computed on; use "
                             f"int(np.ceil(...))")

    # 7a3. EVERY training cutoff purges by the delay AND the horizon.
    #
    #      This is the paper's central design claim - "the training window
    #      stops delta + h days before each forecast origin, so no label used
    #      to fit a model had resolved while the stale mark was still current"
    #      - and leakage here would be the most damaging defect the paper
    #      could have, because every number in it would be inflated and no
    #      figure check could see it: a leaky model reproduces its own output
    #      byte for byte.  Nothing checked it until a sweep did, across all
    #      sixty-nine labs and thirty-eight cutoff sites.
    #
    #      The canonical form is `cut = t - delta - H`, and the sweep found
    #      twenty-five of those plus thirteen variants, every one of which is
    #      legitimate: a benchmark forecaster uses no delayed feature and needs
    #      only the horizon purge, a masked-training arm cuts at the LONGEST
    #      delay which is stricter, and one arm is deliberately optimistic and
    #      labelled as such.  Each exception is listed below with its reason,
    #      because an exception without a reason is how a leak would enter.
    _CUT_OK = {
        # benchmark forecasters: a trailing mean of RESOLVED labels, using no
        # delayed own feature, so the horizon purge is the whole requirement.
        # These cannot inflate the substitution rate in any case: the rate is a
        # ratio of two skill differences against the SAME benchmark, which
        # cancels it exactly - the companion proves that in exact arithmetic.
        ("lab05_robustness.py", "t - horizon"): "bench_mean/bench_var",
        ("lab10_loss_scale.py", "t - H"): "bench_nat, the natural-scale benchmark",
        # stricter than canonical: purge at the LONGEST delay in the grid
        ("lab29_single_regime.py", "t - max(DELAYS) - L.HORIZON"): "longest delay",
        ("lab32_masked_training.py", "t - cut_back - H"): "cut_back = max(train_grid)",
        # the one deliberately optimistic arm, which exists to price the
        # conservative cut above it and is labelled OPTIMISTIC in the source
        ("lab32_masked_training.py", "t - min(train_grid) - H"):
            "declared optimistic, priced against the conservative cut",
    }
    _CUTPAT = re.compile(r"^\s*(?:cut|stop|end|hi)\s*=\s*(.+)$")
    _DELAY_H = re.compile(r"-\s*\w+\s*-\s*(?:H|h|HORIZON|HOR|H_M|H_Q|"
                          r"L\.HORIZON)\b")
    _nsites = 0
    for _nm in sorted(f for f in os.listdir(os.path.join(HERE, "labs"))
                      if f.endswith(".py")):
        for _i, _ln in enumerate(
                open(os.path.join(HERE, "labs", _nm),
                     encoding="utf-8").read().splitlines(), 1):
            if _ln.lstrip().startswith("#"):
                continue
            _m = _CUTPAT.match(_ln)
            if not _m:
                continue
            _rhs = _m.group(1).split("#")[0].strip()
            if not re.search(r"\bt\b", _rhs):
                continue
            _nsites += 1
            if _DELAY_H.search(_rhs):
                continue
            if (_nm, _rhs) in _CUT_OK:
                continue
            fails.append(
                f"labs/{_nm}:{_i} sets a training cutoff of `{_rhs}`, which "
                f"does not purge by both a delay and the horizon and is not "
                f"one of the {len(_CUT_OK)} declared exceptions. If it is a "
                f"benchmark or a stricter cut, add it to _CUT_OK in "
                f"verify_repo.py with the reason; if it is not, a label that "
                f"resolved after the forecaster's cutoff is in its training "
                f"set and every figure downstream of it is inflated")
    if _nsites < 30:
        fails.append(f"the training-cutoff sweep found only {_nsites} sites, "
                     f"where the labs hold about thirty-eight; the pattern has "
                     f"stopped matching and the check is no longer checking")

    # 7a4. the OTHER half of the real-time claim: a foreign close enters only
    #      from a timestamp strictly preceding the target's own.  That rule
    #      lives in exactly one place, lab05.build's `lag` dictionary, and
    #      every lab's panel routes through it; a sweep of the fifty-four labs
    #      that touch a peer block found no second implementation.  Two things
    #      are checked, because both are one character from being wrong.
    _l05 = open(os.path.join(HERE, "labs", "lab05_robustness.py"),
                encoding="utf-8").read()
    #      (i) the comparison must be STRICT.  Under `<=` a peer closing at the
    #      same instant as the target would be admitted as same-day, which is
    #      simultaneous information and not admissible.
    if not re.search(r"lag\s*=\s*\{[^}]*CLOSE_UTC\[p\]\s*<\s*"
                     r"CLOSE_UTC\[target\]", _l05, re.S):
        fails.append("labs/lab05_robustness.py no longer decides peer "
                     "admissibility with a STRICT CLOSE_UTC[p] < "
                     "CLOSE_UTC[target]; under <= a peer closing at the same "
                     "instant as the target would be admitted as same-day, "
                     "which is simultaneous information")
    #      (ii) no lab may undo that shift except the one that exists to price
    #      it.  lab52 builds an INADMISSIBLE block deliberately and prints it
    #      beside the admissible one; a forward shift anywhere else is
    #      lookahead, and a forward shift is what undoing the rule looks like.
    _FWD = re.compile(r"\.shift\(\s*-\d+\s*\)")
    _FWD_OK = {"lab52_compressed_foreign_options.py"}
    for _nm in sorted(f for f in os.listdir(os.path.join(HERE, "labs"))
                      if f.endswith(".py")):
        _src = open(os.path.join(HERE, "labs", _nm), encoding="utf-8").read()
        for _i, _ln in enumerate(_src.splitlines(), 1):
            if _ln.lstrip().startswith("#") or not _FWD.search(_ln):
                continue
            if _nm in _FWD_OK and "undo the admissibility shift" in _ln:
                continue
            fails.append(
                f"labs/{_nm}:{_i} applies a FORWARD shift, {_FWD.search(_ln).group(0)}, "
                f"which moves a series back in time. On a feature that is "
                f"lookahead and on a peer column it undoes the admissibility "
                f"rule. The only permitted one is lab52's declared "
                f"inadmissible arm, which must carry the comment 'undo the "
                f"admissibility shift' on the same line")

    # 7b. every block_for() call is reachable.  The truncation length is a
    #     function now, owned by lab02 and reached through whatever alias a
    #     lab imports it under, and verify_paper.py checks that the delay
    #     loops ask for it.  What no check there can see is whether the ALIAS
    #     exists: `L.block_for(d)` inside a function body is valid Python in a
    #     file that never imports anything as L, imports cleanly, and raises
    #     NameError only when that line is finally reached - which for a lab
    #     forty minutes into a full run is an expensive way to find out.  That
    #     is not hypothetical: lab04 was edited to call L.block_for while
    #     having no `as L` import at all.
    _labdir = os.path.join(HERE, "labs")
    for _nm in sorted(f for f in os.listdir(_labdir)
                      if re.fullmatch(r"lab\d+[a-z]?_.*\.py", f)):
        _src = open(os.path.join(_labdir, _nm), encoding="utf-8").read()
        try:
            _tr = ast.parse(_src)
        except SyntaxError:
            continue                      # verify_paper.py reports the parse
        # Read from the SYNTAX TREE, not the text.  Stripping "#" comments was
        # not enough: the docstring that explains the rule naturally writes
        # "See lab02.block_for()", and matching that reported two labs whose
        # code was correct.  A checker that cries wolf gets switched off.
        _aliases, _bare = set(), False
        for _n in ast.walk(_tr):
            if isinstance(_n, ast.Attribute) and _n.attr == "block_for" \
                    and isinstance(_n.value, ast.Name):
                _aliases.add(_n.value.id)
            elif isinstance(_n, ast.Name) and _n.id == "block_for":
                _bare = True
        for _al in sorted(_aliases):
            # A trailing "# noqa: E402" is how every lab in this repository
            # writes a late import, so the alias may be followed by a comment.
            # Without that the gate reported a NameError in a file whose import
            # was correct, the moment a lab first called L.block_for().
            if not re.search(r"^\s*import\s+lab\w+\s+as\s+" + re.escape(_al)
                             + r"\s*(?:#.*)?$", _src, re.M):
                fails.append(f"{_nm} calls {_al}.block_for() but never imports any "
                             f"lab module as {_al}, so that line raises NameError "
                             f"the moment it is reached")
        if _bare and not re.search(r"^def block_for\b", _src, re.M) and \
                not re.search(r"^\s*(?:from\s+lab\w+\s+import\s+[^\n]*"
                              r"\bblock_for\b|block_for\s*=)", _src, re.M):
            fails.append(f"{_nm} calls block_for() unqualified without defining or "
                         f"importing it")

        # 7c. a lab that RESAMPLES at block_for(delta) may not announce a flat
        #     block in its header.  Four labs did: they printed
        #     "block bootstrap, block 40" and then passed L.block_for(d) to
        #     every call beneath it.  Nothing here could see the contradiction,
        #     and the companion note's Table 7 caption was written from one of
        #     those headers and told readers the note's own tables used a flat
        #     block of 40, which no table in it did.  A header that contradicts
        #     the code under it is not a cosmetic problem: it is where a wrong
        #     number in a paper comes from.
        _uses_rule = any(
            isinstance(_n, ast.Call)
            and ((isinstance(_n.func, ast.Attribute) and _n.func.attr == "block_for")
                 or (isinstance(_n.func, ast.Name) and _n.func.id == "block_for"))
            for _n in ast.walk(_tr))
        if _uses_rule:
            for _m in re.finditer(r'block bootstrap, block \{(\w+(?:\.\w+)?)\}',
                                  _src):
                fails.append(f"{_nm} resamples at block_for(delta) but its header "
                             f"announces a flat block of {{{_m.group(1)}}}; print the "
                             f"rule and its range, not its first value")

    # 7d. build_variants.py may not carry a result interval as a literal.
    #     Each of its three journal abstracts used to keep its own copy of the
    #     conditional rate, the middle tercile and the headline -- four copies
    #     of each figure in this repository -- and they drifted: when the
    #     inference moved to fixed-b critical values the paper's headline
    #     became one interval and all three abstracts here kept the old one,
    #     invisible because the abstract check required only the bare "71%".
    #     They read the labs now, and a literal is how that would be undone.
    _bv = os.path.join(HERE, "build_variants.py")
    if os.path.isfile(_bv):
        _bvsrc = open(_bv, encoding="utf-8").read()
        # only inside the abstract constants, and only the "NN% [a, b]" shape:
        # the file legitimately contains intervals in comments explaining this
        # very history, and a ban that cannot tell the two apart would be a
        # check that cries wolf.
        for _m in re.finditer(r'^([A-Z]+_ABSTRACT) = """(.*?)"""', _bvsrc,
                              re.S | re.M):
            _lits = re.findall(r"\d+% \[\s*[+-]?\d+\s*,\s*[+-]?\d+\s*\]",
                               _m.group(2))
            if _lits:
                fails.append(f"build_variants.py: {_m.group(1)} carries the "
                             f"interval(s) {', '.join(_lits)} as a literal; use a "
                             f"placeholder filled from the labs, as the other "
                             f"figures in that abstract are")

    # 7e. the README must not claim the verifier checks "every figure".  It
    #     said that for a long time, and the claim was false in a specific and
    #     consequential way: bare prose percentages are not individually
    #     pinned.  An overclaim in a replication README is worse than a gap,
    #     because it tells a reader not to look.
    _rd = os.path.join(HERE, "README.md")
    if os.path.isfile(_rd):
        _rdsrc = open(_rd, encoding="utf-8").read()
        # The README is wrapped prose, so a required phrase can straddle a
        # newline: "a check may\nnot contain a figure it did not read" failed a
        # literal search of the file while being present in the text.  Both the
        # raw file and a whitespace-collapsed copy are searched, and a phrase
        # found in either counts.
        _rdflat = " ".join(_rdsrc.split())
        for _bad in ("check every figure in all three documents",
                     "checks every figure in the paper",
                     "every figure is pinned"):
            if _bad in _rdsrc or " ".join(_bad.split()) in _rdflat:
                fails.append(f"README.md claims '{_bad}', which overstates what "
                             f"verify_paper.py covers; the coverage section says "
                             f"what it actually does")
        # The README's own account of the residual must survive the audit that
        # rewrote it: this list used to require "Not fully covered:", which the
        # section still says in longer form, and it did NOT require the README
        # to withdraw its claim that nothing in the uncovered prose was wrong.
        # A cold pass found forty errors in exactly that prose, so the
        # withdrawal is now the thing a checker insists on.
        for _bad2 in ("None of them is wrong",
                      "each traces to a lab output and every verifier passes"):
            if _bad2 in _rdsrc or " ".join(_bad2.split()) in _rdflat:
                fails.append(f"README.md still says '{_bad2}' about the prose no "
                             f"check reads; an audit found forty errors there, so "
                             f"the claim is withdrawn rather than softened")
        for _need in ("asks the converse",
                      "Not fully covered, and this is where the errors were:",
                      "a check may not contain a figure it did not read",
                      "no check reads it",
                      "twenty-four of twenty-four"):
            if _need not in _rdsrc and " ".join(_need.split()) not in _rdflat:
                fails.append(f"README.md no longer says '{_need}', so its account "
                             f"of what the verifier covers is incomplete")

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
            ("why-a-simple-threshold-rule-is-hard-to-beat.html",
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

    # 10. Every variant folder must say which of its files go to the journal.
    #     Each one holds five documents and two of them name the author, with
    #     "stale-mark-internet-appendix.pdf" sitting beside
    #     "internet-appendix-anonymous.pdf".  Under submission-day pressure the
    #     difference between those two filenames is one wrong click and an
    #     anonymity that cannot be restored.  build_variants.py writes an upload
    #     manifest into each folder; this is what makes its absence a failure
    #     rather than something noticed afterwards.
    _vd = os.path.join(HERE, "papers", "variants")
    if os.path.isdir(_vd):
        for _v in sorted(os.listdir(_vd)):
            if not os.path.isdir(os.path.join(_vd, _v)):
                continue
            _man = os.path.join(_vd, _v, "UPLOAD-THESE.txt")
            if not os.path.isfile(_man):
                fails.append(f"papers/variants/{_v}/ has no UPLOAD-THESE.txt, so "
                             f"nothing in the folder says which of its five "
                             f"documents go to the journal")
                continue
            _mt = open(_man, encoding="utf-8").read()
            for _need in ("title-page.pdf", "manuscript-anonymous.pdf",
                          "internet-appendix-anonymous.pdf"):
                if _need not in _mt:
                    fails.append(f"papers/variants/{_v}/UPLOAD-THESE.txt does not "
                                 f"name {_need}, which is one of the files that "
                                 f"must go up")
            if "DO NOT UPLOAD" not in _mt:
                fails.append(f"papers/variants/{_v}/UPLOAD-THESE.txt does not warn "
                             f"against the two identified copies in the same folder")

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

    # 10b. every figure in a README lab row must come from THAT lab's output.
    #
    # The README's table of sixty-nine labs carries about a hundred and twenty
    # figures and had no check of any kind behind it.  Fourteen of them were
    # wrong: lab48's three intervals, lab45's two, lab50's one, lab42's
    # persistence and ceiling, lab44's whole Giacomini-White range, lab40's
    # upper bound, lab46's residual, lab57's de-smoothed cost - with its SIGN,
    # which was the point of the sentence - and lab68's null mean, standard
    # deviation and largest count, the same stale copy that had been found
    # twice already in the manuscript's condensation and once in the
    # abstracts.  Several had been correct when written and were made wrong by
    # a later repair to the lab, which is the drift this repository exists to
    # prevent and had been allowing in the file a reader opens first.
    #
    # The test is ROW-SCOPED on purpose.  Run globally - does this number
    # appear in any lab output - it passes eleven of the fourteen, because
    # with sixty-nine files of printed tables almost any four-digit decimal
    # occurs somewhere.  That weaker version was written first and reported
    # three problems instead of fourteen.
    # Integers count too.  The first version of this matched decimals and
    # percentages only, and "the largest being 64" - where lab68 prints 55 -
    # walked straight through it, which is the figure that had already been
    # wrong in three other places.  Extending it to two-to-four digit
    # integers found two more real errors, lab62's 79 and 74 against its own
    # 77 and 72, at the cost of two false alarms, both citation years, which
    # the second pattern excludes by the "(Author YEAR)" shape around them.
    _RDFIG = re.compile(r"(?<![\d.\w])\d{1,4}\.\d{1,4}(?![\d.])"
                        r"|(?<![\d.\w])\d{1,3}%(?!\w)"
                        r"|(?<![\d.,\w])\d{2,4}(?![\d.,%])")
    _RDYEAR = re.compile(r"(?:and |&|, )?\(?(?:19|20)\d\d\)?[),]?\s*$")

    def _traceable(val, out):
        """val appears in out, verbatim or as out's own figure rounded."""
        bare = val.rstrip("%")
        if bare in out:
            return True
        # a paper that writes 3.2 where the lab prints 3.19 is rounding, not
        # drifting; accepted only at the figure's OWN precision, against a
        # number from the same output.
        dp = len(bare.split(".")[1]) if "." in bare else 0
        for _m in re.finditer(r"(?<![\d.\w])\d{1,4}\.\d{1,6}(?![\d.])", out):
            try:
                if f"{float(_m.group(0)):.{dp}f}" == bare:
                    return True
            except ValueError:
                pass
        return False

    for _line in txt.splitlines():
        _m = re.match(r"\|\s*`(lab\w+)\.py`\s*\|(.*)\|\s*$", _line)
        if not _m:
            continue
        _lab, _body = _m.group(1), _m.group(2)
        _lo = os.path.join(HERE, "expected_output", _lab + ".txt")
        if not os.path.isfile(_lo):
            continue          # rule 1 above already reports a missing output
        _out = open(_lo, encoding="utf-8").read()
        for _f in _RDFIG.finditer(_body):
            checks_rd = _f.group(0)
            # a year in a citation - "(Kiefer and Vogelsang 2005)", "Shao's
            # (2010)" - is a reference, not a figure from the lab
            if (re.fullmatch(r"(?:19|20)\d\d", checks_rd)
                    and _RDYEAR.search(_body[:_f.end()])):
                continue
            if not _traceable(checks_rd, _out):
                fails.append(
                    f"the README's {_lab} row quotes {checks_rd}, which that "
                    f"lab's own output does not produce  "
                    f"...{' '.join(_body[max(0, _f.start() - 60):_f.end() + 25].split())}...")

    # 10c. the README's multiplicity sentence, pinned claim by claim.
    #
    # The row-scoped test above catches a figure the named lab no longer
    # produces anywhere.  It cannot catch one that still occurs in that
    # output in a DIFFERENT ROLE, and the stalest figure in this file was
    # exactly that: the README said the null's largest count in two thousand
    # draws was 64, where lab68 says 55 - and "64" does occur in lab68's
    # output, twice, as the size of two of the secondary grids.  So the one
    # sentence that has now been found stale in four separate documents gets
    # needles instead of a sweep, read from the lab's own summary line.
    _l68 = os.path.join(HERE, "expected_output",
                        "lab68_joint_multiplicity.txt")
    if os.path.isfile(_l68):
        _l68t = " ".join(open(_l68, encoding="utf-8").read().split())
        _s = re.search(r"what the null actually is, rule A: mean ([\d.]+), "
                       r"sd ([\d.]+), 95th percentile (\d+)", _l68t)
        _p = re.search(r"p_joint = [\d.]+, bounded above by [\d.]+, the null's "
                       r"largest count in (\d+) draws being (\d+)", _l68t)
        # the calibrated rule, which is the margin the README must now quote
        _rc = re.search(r"on rule C, the only correctly sized one: (\d+) "
                        r"observed, null mean ([\d.]+), 95th percentile (\d+), "
                        r"p_joint = ([\d.]+)", _l68t)
        _o = re.search(r"observed significant cells (\d+)", _l68t)
        _c = re.search(r"cells (\d+) calendar groups \d+", _l68t)
        checks_m = 0
        if not (_s and _p and _o and _c and _rc):
            fails.append("lab68 no longer prints the summary lines the "
                         "README's multiplicity sentence is read against")
        else:
            _rd1 = " ".join(txt.split())
            for _label, _need in (
                    ("null mean and sd",
                     f"mean {_s.group(1)} and sd {_s.group(2)}"),
                    ("95th percentile", f"95th percentile of {_s.group(3)}"),
                    ("largest count", f"the largest being {_p.group(2)}"),
                    ("replications", f"in any of 2,000 replications"),
                    ("observed count",
                     f"set {_o.group(1)} nominally significant"),
                    ("pool size", f"{_c.group(1)} secondary cells"),
                    ("the calibrated rule's count and p",
                     f"{_rc.group(1)} against a null mean of "
                     f"{float(_rc.group(2)):.1f} and p = {_rc.group(4)}")):
                checks_m += 1
                if _need not in _rd1:
                    fails.append(f"the README's multiplicity sentence does not "
                                 f"carry lab68's {_label}: expected "
                                 f"{_need!r}")

    # 11a. no shipped script may name an absolute path.
    #
    # verify_paper.py shipped with two lines writing debug output to
    # /tmp/claude-0/, a session-scratch directory on the machine it was
    # written on.  Every reader's run would have died there with
    # FileNotFoundError, and on Windows the path cannot exist at all - so the
    # verifier this repository is built around was unrunnable for everyone
    # except the author, and nothing noticed, because the directory exists
    # wherever it was being run.  Even the clone test missed it: the clone was
    # built on the same machine.
    #
    # The rule is flat and mechanical: a string literal in a shipped script
    # that looks like an absolute filesystem path is a failure.  Paths are
    # built from HERE and os.path.join, which is what every other line here
    # already does.  URLs are not paths and are left alone.
    # Anchored on the WHOLE string, with no spaces allowed and at least two
    # path segments one of which contains a letter.  The first version matched
    # any literal beginning with "/", which in a repository full of printed
    # tables meant "/0.94  lag10 " and "/UPLOAD-THESE.txt does not name " -
    # six false alarms against two real ones, and a gate that cries wolf is a
    # gate that gets switched off.
    _ABS = re.compile(r"^(?:(?:/[A-Za-z0-9_.\-]+){2,}/?"
                      r"|[A-Za-z]:[\\/][A-Za-z0-9_.\\/\-]+"
                      r"|~/[A-Za-z0-9_.\-/]+)$")
    for rel in shipped:
        if not rel.endswith(".py"):
            continue
        try:
            _tree = ast.parse(open(os.path.join(HERE, rel),
                                   encoding="utf-8").read())
        except SyntaxError as _e:
            fails.append(f"{rel} does not parse: {_e}")
            continue
        for _node in ast.walk(_tree):
            if not (isinstance(_node, ast.Constant)
                    and isinstance(_node.value, str)):
                continue
            _v = _node.value
            if (_ABS.match(_v) and "://" not in _v
                    and re.search(r"[A-Za-z]", _v)):
                fails.append(f"{rel} line {_node.lineno} names the absolute "
                             f"path {_v!r}; a shipped script must "
                             f"build paths from HERE, or it runs only on the "
                             f"machine it was written on")

    # 11b. data/illiquid/PROVENANCE.md, which was the ONE shipping file no
    # gate read.
    #
    # A coverage map over all 175 shipping files found exactly two that
    # nothing here opens: LICENSE, which carries no claim, and this one,
    # which carries eight SHA-256 digests, four byte counts, three row counts
    # and two figures from lab54 - every one of them a second copy of
    # something stored elsewhere.  They all agreed when the map was drawn.
    # They agreed because the file was written and not touched again, which
    # is the same reason the condensation's figures agreed for a while.
    _pv = os.path.join(HERE, "data", "illiquid", "PROVENANCE.md")
    checks_pv = 0
    if not os.path.isfile(_pv):
        fails.append("data/illiquid/PROVENANCE.md is gone; the illiquid "
                     "series would ship with no account of where they came "
                     "from")
    else:
        _pvt = open(_pv, encoding="utf-8").read()
        _man = {_r["path"]: _r for _r in csv.DictReader(
            open(os.path.join(HERE, "data", "MANIFEST.tsv"), encoding="utf-8"),
            delimiter="\t")}
        # the national series' table: file, FRED id, rows, bytes, digest
        _seen = 0
        for _m in re.finditer(r"\|\s*`(\w+_fred\.csv)`\s*\|\s*(\w+)\s*\|\s*"
                              r"(\d+)\s*\|\s*(\d+)\s*\|\s*`([0-9a-f]{64})`",
                              _pvt):
            _f, _fid, _rows, _bytes, _sha = _m.groups()
            _r = _man.get("illiquid/" + _f)
            _seen += 1
            if not _r:
                fails.append(f"PROVENANCE.md tabulates {_f}, which the "
                             f"manifest does not list")
                continue
            for _label, _got, _want in (("rows", _rows, _r["rows"]),
                                        ("bytes", _bytes, _r["bytes"]),
                                        ("SHA-256", _sha, _r["sha256"])):
                checks_pv += 1
                if _got != _want:
                    fails.append(f"PROVENANCE.md gives {_f} {_label} {_got}; "
                                 f"the manifest says {_want}")
        checks_pv += 1
        if _seen != 3:
            fails.append(f"PROVENANCE.md's table of national series has "
                         f"{_seen} rows, not the three the manifest holds")
        # the metro panel, described in prose rather than in the table
        _mp = _man["illiquid/CaseShiller_metro20_NSA.csv"]
        for _label, _pat, _want in (
                ("SHA-256", r"`CaseShiller_metro20_NSA\.csv`.{0,500}?"
                            r"SHA-256 `([0-9a-f]{64})`", _mp["sha256"]),
                ("bytes", r"(\d+) bytes", _mp["bytes"]),
                ("months", r"(\d+) months \(1987-01 to 2026-06\)",
                 _mp["rows"])):
            checks_pv += 1
            _m = re.search(_pat, _pvt, re.S)
            if not _m:
                fails.append(f"PROVENANCE.md no longer states the metro "
                             f"panel's {_label} in the form this reads")
            elif _m.group(1) != _want:
                fails.append(f"PROVENANCE.md gives the metro panel's {_label} "
                             f"as {_m.group(1)}; the manifest says {_want}")
        # and the two lab54 figures it quotes, from lab54's own output
        _l54 = open(os.path.join(HERE, "expected_output",
                                 "lab54_monthly_feasibility.txt"),
                    encoding="utf-8").read()
        _wid = dict(re.findall(r"^\s+(\d+)%\s+(\d+)\s+\d+\s*$", _l54, re.M))
        # matched against the file with its line wrapping intact, so \s+
        # rather than a literal space: "needs about 121\nmonth-marks" is how
        # it is actually stored, and a space-only pattern missed it.
        for _pt, _pat in (("40", r"a\s+40-point\s+interval\s+needs\s+about"
                                 r"\s+(\d+)\s+month-marks"),
                          ("30", r"a\s+30-point\s+interval\s+needs\s+(\d+)")):
            checks_pv += 1
            _m = re.search(_pat, _pvt)
            if not _m:
                fails.append(f"PROVENANCE.md no longer states lab54's "
                             f"{_pt}-point requirement in the form this reads")
            elif _wid.get(_pt) != _m.group(1):
                fails.append(f"PROVENANCE.md says a {_pt}-point interval needs "
                             f"{_m.group(1)} month-marks; lab54 prints "
                             f"{_wid.get(_pt)}")

    # 12. the clone test, run rather than asserted.
    #
    # The README promises that a clone missing contact.txt and submission/
    # "builds and verifies cleanly - the checks that read them say so and pass
    # rather than failing for everyone but the author".  That sentence was
    # false for a year and nothing noticed, because every run of these scripts
    # happened in a working copy that HAS both files.  build_variants.py
    # stopped the whole build over a missing cover letter, so the one reader
    # the promise was written for - a replicability reviewer with a fresh
    # clone - was the only one who would ever see it fail.
    #
    # So the promise is now tested the only way it can be: copy exactly the
    # shipping set into a temporary tree, with neither withheld file, and run
    # the two scripts there.  It costs about half a minute and it is the one
    # check in this file that exercises the repository instead of reading it.
    if "--clone" in argv:
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp(prefix="clone-test-")
        try:
            for rel in shipped:
                dst = os.path.join(tmp, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(os.path.join(HERE, rel), dst)
            for _name, _args in (
                    ("verify_paper.py",
                     ["papers/source/what-substitutes-for-a-stale-mark.html",
                      "papers/source/"
                      "why-a-simple-threshold-rule-is-hard-to-beat.html"]),
                    ("build_variants.py", [])):
                checks_done = subprocess.run(
                    [sys.executable, os.path.join(tmp, _name)] + _args,
                    capture_output=True, text=True, cwd=tmp)
                if checks_done.returncode != 0:
                    fails.append(
                        f"the README says a clone builds and verifies cleanly, "
                        f"but {_name} exits {checks_done.returncode} in one: "
                        f"{(checks_done.stdout + checks_done.stderr).strip()[-400:]}")
            # and the two withheld files must really be absent from it
            for _w in ("contact.txt", "submission"):
                if os.path.exists(os.path.join(tmp, _w)):
                    fails.append(f"the clone test is not testing a clone: {_w} "
                                 f"is in the shipping set")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # 15. No input may be dated on a day its own exchange was shut.
    #
    #     This is the check the repository did not have.  Every timing rule here
    #     compares one timestamp with another - a close in UTC against the
    #     target's close, a peer shifted back a day, VIX entering at t-1 - and
    #     a row dated on a Sunday satisfies all of them.  Thirty Nikkei futures
    #     prints and one 26%-wrong Hang Seng print sat in the panel for two
    #     years because of it, each corrupting about a year of that target's
    #     decision variable through the trailing median.
    #
    #     It fails in BOTH directions, which is the part that matters:
    #       - a market not exempted whose weekend rows survive means the gate in
    #         lab05.load_index has been removed or bypassed;
    #       - a market exempted whose weekend rows do NOT look like cash means
    #         somebody widened WEEKEND_SESSIONS to make an error go away.
    #     The second is the likelier mistake and the one a passing gate hides.
    #
    #     lab72_trading_day_audit prints the full audit; this is the refusal.
    #     The folder comes from --data or from the repository's own data/, and
    #     from nowhere else.  A first version of this block carried the author's
    #     own absolute data path as a third fallback, which the absolute-path
    #     gate thirty lines above caught within the minute - a check that only
    #     runs on one machine is the defect class this file already covers, and
    #     writing a fresh instance of it into the same file is not an excuse.
    _data = None
    for _cand in ([argv[argv.index("--data") + 1]] if "--data" in argv else []) + \
                 [os.path.join(HERE, "data")]:
        if os.path.isdir(_cand):
            _data = _cand
            break
    if _data is None:
        print("  skip  trading-day gate: no data folder present (clone)")
    else:
        sys.path.insert(0, os.path.join(HERE, "labs"))
        try:
            import lab05_robustness as _L
        except Exception as _e:
            fails.append(f"the trading-day gate could not be checked: {_e}")
            _L = None
        if _L is not None:
            _seen = 0
            for _t in _L.CLOSE_UTC:
                if not _L.files_for(_t, _data):
                    continue
                _seen += 1
                _m = _L.load_index(_t, _data)
                _we = _m.loc[_m["date"].dt.dayofweek >= 5]
                if not len(_we):
                    continue
                if _t not in _L.WEEKEND_SESSIONS:
                    fails.append(
                        f"{_t} keeps {len(_we)} weekend-dated rows after "
                        f"load_index, and {_t} is not in WEEKEND_SESSIONS - the "
                        f"trading-day gate is not being applied")
                    continue
                # exempted: the rows must actually look like cash sessions
                _wd = _m.loc[_m["date"].dt.dayofweek < 5]
                def _sig(d):
                    import numpy as _np
                    return (float(_np.isclose(d["close"] % 1, 0).mean()),
                            float(((d["high"] - d["low"]) / d["close"]).mean()))
                _rw, _gw = _sig(_we), _sig(_wd)
                if _rw[0] > 4 * max(_gw[0], 0.01) and _rw[1] < 0.5 * _gw[1]:
                    fails.append(
                        f"{_t} is exempted in WEEKEND_SESSIONS, but its "
                        f"{len(_we)} weekend rows do not look like cash "
                        f"sessions: {_rw[0]:.1%} whole-number closes against "
                        f"{_gw[0]:.1%} on weekdays and a {_rw[1]:.3%} mean range "
                        f"against {_gw[1]:.3%}. That signature is another "
                        f"instrument, so exempting {_t} hides the defect "
                        f"instead of fixing it")
            if _seen == 0:
                print("  skip  trading-day gate: no index files under "
                      f"{_data}")

    # ---- gate 16: a lab that reads another lab's output must run after it ---
    #
    # Five labs read a sibling's STORED output rather than recomputing its
    # result, which is right: a figure that belongs to lab06 should not get a
    # second home in lab07.  But it makes run_all's ORDER load-bearing, and
    # nothing said so.  A regeneration that ran lab07 before lab06 left lab07
    # quoting lab06's pre-correction figures, and verify_paper reported it as
    # lab07 disagreeing with lab06 - which was true, and whose cause was an
    # order nobody had written down.  It is written down here now.
    _order = []
    _ra = open(os.path.join(HERE, "run_all.py"), encoding="utf-8").read()
    for _m in re.finditer(r'\("(lab\d+[a-z]?_[a-z_0-9]+)\.py"', _ra):
        _order.append(_m.group(1))
    _pos = {name: i for i, name in enumerate(_order)}
    _labdir = os.path.join(HERE, "labs")
    for _f in sorted(os.listdir(_labdir)):
        if not re.fullmatch(r"lab\d+[a-z]?_.*\.py", _f):
            continue
        _me = _f[:-3]
        _src = open(os.path.join(_labdir, _f), encoding="utf-8").read()
        if "expected_output" not in _src:
            continue
        for _dep in sorted(set(re.findall(r"(lab\d+[a-z]?_[a-z_0-9]+)\.txt",
                                          _src))):
            if _dep == _me:
                continue
            if _me not in _pos:
                fails.append(f"{_me} reads {_dep}'s stored output but is not in "
                             f"run_all's ORDER at all, so nothing can guarantee "
                             f"it runs after it")
            elif _dep not in _pos:
                fails.append(f"{_me} reads {_dep}'s stored output but {_dep} is "
                             f"not in run_all's ORDER")
            elif _pos[_dep] > _pos[_me]:
                fails.append(f"{_me} reads {_dep}'s stored output, and run_all "
                             f"runs {_dep} AFTER it (positions {_pos[_dep]} and "
                             f"{_pos[_me]}), so a full rerun leaves {_me} "
                             f"quoting the previous vintage")

    print(f"verify_repo: {len(shipped)} files ship, {n_labs} labs, "
          f"{n_gates} registry gates, {len(rows)} table rows"
          + ("" if "--clone" in argv else "   (--clone also builds one)"))
    for f in fails:
        print("  FAIL  " + f)
    if not fails:
        print("  the README agrees with the repository it describes")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
