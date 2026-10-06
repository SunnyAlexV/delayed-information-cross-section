"""derive_table_map.py - work out which lab row every result-table row came from.

    python derive_table_map.py [--old <dir>]      writes papers/TABLE_MAP.tsv

WHY THIS FILE EXISTS
--------------------
The result tables in both documents were maintained by hand.  Every time a lab
moved, a hundred-odd cells went stale, and the only thing that found them was
verify_paper reporting a hundred-odd failures that then had to be patched one at
a time.  That is work which has to be redone on the next change, and it has been
redone several times already.

This project does not maintain that kind of thing by hand anywhere else.
CONVENTIONS.tsv is generated from lab05's own constants so it cannot disagree
with the code that applies them.  build_variants fills figures from lab output
and has a gate that FAILS if a result figure is typed into generated prose.  The
result tables simply predate that machinery.

So: this file derives, once, the map from every paper table cell to the lab
token it is a copy of.  build_tables.py then rewrites those cells from the labs
on every build, and the whole class of defect goes away rather than being
cleared one instance at a time.

HOW THE DERIVATION WORKS, AND WHY IT IS DONE AGAINST THE OLD OUTPUTS
--------------------------------------------------------------------
A paper row is matched to a lab row by its SEQUENCE OF NUMBERS, as an ordered
subsequence of that lab row's tokens.  A sequence of six or eight figures is
close to unique across two thousand lab rows, where any single figure is not -
which is the same row-scoping lesson the README figure gate already encodes.

The match is made against the lab outputs AS THEY WERE WHEN THE PAPER WAS LAST
CONSISTENT, not against the current ones, because that is the state the paper's
cells are copies of.  Pass that directory with --old.  Cells already corrected
by hand will match the CURRENT value instead, so both are accepted.

WHAT IT REFUSES TO DO
---------------------
It records a row only when exactly one lab row matches.  An ambiguous row is
left out of the map and reported, and build_tables.py then leaves that row
alone.  A map that guessed would be worse than a map with holes: a wrong entry
would rewrite a correct figure on every future build, silently and for ever.
The coverage number this prints is therefore a floor, and the rows it skips stay
hand-maintained until someone gives them a key.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS = ["what-substitutes-for-a-stale-mark.html",
          "stale-mark-internet-appendix.html",
          "why-a-simple-threshold-rule-is-hard-to-beat.html"]
NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")
CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
TR = re.compile(r"<tr[^>]*>.*?</tr>", re.S | re.I)
TABLE = re.compile(r"<table\b.*?</table>", re.S | re.I)


def _render(tok):
    """A lab token in the paper's notation - must match build_tables.render."""
    # The TEMPLATE carries the paper's own notation - its brackets, its per
    # cent sign, its thousands separator - so the token must arrive bare or the
    # two compose into '29.7%%'.  That doubling silently failed the map gate on
    # every percentage cell in the paper, which is most of the rate columns.
    tok = tok.strip("[],%").replace(",", "")
    return "&minus;" + tok[1:] if tok.startswith("-") else tok


def norm(s):
    """Paper markup to plain text, with every dash spelling folded to '-'."""
    s = re.sub(r"<[^>]+>", "", s)
    return (s.replace("&minus;", "-").replace("−", "-")
             .replace("&nbsp;", " ").replace("&ndash;", "-").strip())


def nums(s):
    """The numbers in a cell, sign-normalised and without a leading '+'."""
    return [x.lstrip("+") for x in NUM.findall(norm(s))]


def lab_rows(folder):
    """Every numeric row of every lab output, as (file, block, key, tokens).

    Blocks are separated by the labs' own ruler lines, so a row's position is
    recorded relative to its block rather than to the file, and inserting a
    paragraph above a table does not move it.
    """
    out = []
    seen_keys = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".txt"):
            continue
        blk = 0
        for ln in open(os.path.join(folder, fn), encoding="utf-8",
                       errors="replace"):
            st = ln.strip()
            if st and set(st) <= set("=-") and len(st) > 20:
                blk += 1
                continue
            toks = ln.split()
            if len(toks) >= 3 and len(NUM.findall(ln)) >= 2:
                # A block can hold SEVERAL rows with the same leading label -
                # lab15's block 4 carries two rows keyed '0', one per arm - so
                # the label alone is not an address.  The occurrence index makes
                # it one, and is stable as long as the lab prints its tables in
                # the same order, which is what expected_output pins anyway.
                nth = seen_keys.get((fn, str(blk), toks[0]), 0)
                seen_keys[(fn, str(blk), toks[0])] = nth + 1
                out.append((fn, str(blk), f"{toks[0]}#{nth}", list(toks)))
    # A paper row may be a JOIN of two lab rows.  Table 12 is the clearest
    # case: its columns through "GW z" come from lab44's part A and the foreign
    # increment from its part B, and the paper's own caption says so.  Without
    # this, every joined table stays hand-maintained.  Only rows sharing a file
    # AND a key are joined, so the pairing is the one the paper makes rather
    # than an arbitrary splice of two unrelated tables.
    byk = {}
    for fn, blk, key, toks in out:
        byk.setdefault((fn, key), []).append((blk, toks))
    for (fn, key), lst in byk.items():
        if len(lst) < 2:
            continue
        for i in range(len(lst)):
            for j in range(len(lst)):
                if i == j or lst[i][0] == lst[j][0]:
                    continue
                out.append((fn, f"{lst[i][0]}+{lst[j][0]}", key,
                            lst[i][1] + lst[j][1][1:]))
    return out


def same(paper, lab):
    """Is this lab token the number the paper is quoting?

    Exact first.  Then AT THE PAPER'S OWN PRECISION, because the tables round:
    Section 4.6 prints 71% where lab10 computes 71.5%, and a map that insisted
    on equality would leave every rounded column unmapped and hand-maintained -
    which is the problem this file exists to remove.  Rounding is applied in one
    direction only, lab to paper, so the paper can never be the more precise of
    the two and a genuine disagreement in a shared digit still fails to match.
    """
    lab = lab.strip("[],%").replace(",", "").lstrip("+")
    if lab == paper or lab == paper.lstrip("+"):
        return True
    try:
        dp = len(paper.split(".")[1]) if "." in paper else 0
        return f"{round(float(lab), dp):.{dp}f}" == f"{float(paper):.{dp}f}"
    except ValueError:
        return False


def best_partial(need, hay):
    """The longest ordered run of `need` findable in `hay`, greedily.

    Returns (positions in need, indices in hay).  A paper row is not always a
    copy of ONE lab row: Table 2 interleaves lab10's R-squared pair, then a rate
    and interval computed elsewhere, then lab10's QLIKE pair.  Insisting that a
    whole row come from a single lab row left every interleaved table
    hand-maintained, which is most of them.  Matching the longest run instead
    lets a row be assembled from two sources, with the cells no source covers
    simply left unmapped.
    """
    pos, idx = [], []
    j = 0
    for k, want in enumerate(need):
        jj = j
        while jj < len(hay):
            if same(want, hay[jj]):
                pos.append(k)
                idx.append(jj)
                j = jj + 1
                break
            jj += 1
    return pos, idx


def subseq(need, hay):
    """Indices placing `need` in `hay` in order, or None."""
    idx = []
    j = 0
    for want in need:
        while j < len(hay):
            if same(want, hay[j]):
                idx.append(j)
                j += 1
                break
            j += 1
        else:
            return None
    return idx


def main(argv):
    old = None
    if "--old" in argv:
        old = argv[argv.index("--old") + 1]
    # The map is derived from the paper AS IT WAS when it last agreed with the
    # labs, because that is the state whose cells are copies of lab tokens.  A
    # paper already part-corrected by hand no longer reproduces the old labs,
    # and every candidate would be rejected by the gate below - correctly, but
    # uselessly.  Point --papers at the matching snapshot.
    psrc = os.path.join(HERE, "papers", "source")
    if "--papers" in argv:
        psrc = argv[argv.index("--papers") + 1]
    print(f"  deriving from papers in: {psrc}")
    cur = os.path.join(HERE, "expected_output")
    pools = [("current", lab_rows(cur))]
    if old and os.path.isdir(old):
        pools.insert(0, ("pre-fix", lab_rows(old)))
    print(f"  lab rows: " + ", ".join(f"{n} {len(p)}" for n, p in pools))

    rows_out = []
    seen = amb = miss = rejected = 0
    for pf in PAPERS:
        path = os.path.join(psrc, pf)
        if not os.path.isfile(path):
            continue
        t = open(path, encoding="utf-8").read()
        for ti, tm in enumerate(TABLE.finditer(t)):
            for ri, trm in enumerate(TR.finditer(tm.group(0))):
                cells = CELL.findall(trm.group(0))
                if not cells:
                    continue
                flat, owner = [], []
                for ci, c in enumerate(cells):
                    for v in nums(c):
                        flat.append(v)
                        owner.append(ci)
                if len(flat) < 4:        # too short to identify safely
                    continue
                seen += 1
                # the row's leading label, when it is a bare number (a delay,
                # a tercile index): the lab row must carry the same one
                lab0 = norm(cells[0])
                label = lab0 if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", lab0) else None
                # Match the row against lab rows by the LONGEST ordered run
                # of its figures, and allow a second lab row to cover what the
                # first did not.  Specificity still decides: the pre-fix pool
                # before the current one, single rows before joined pairs.
                assigned = {}          # position in flat -> (src, token index)
                for _, pool in pools:
                    for joined in (False, True):
                        for _round in (0, 1):
                            todo = [k for k in range(len(flat))
                                    if k not in assigned]
                            if len(todo) < 4:
                                break
                            sub = [flat[k] for k in todo]
                            best, cands = 0, []
                            for fn_, blk_, key_, toks_ in pool:
                                if ("+" in blk_) != joined:
                                    continue
                                if label and not any(
                                        label == tk.strip("[],%").replace(",", "")
                                        for tk in (key_,) + tuple(toks_)):
                                    continue
                                pos, ix_ = best_partial(sub, toks_)
                                if len(pos) < 4:
                                    continue
                                if ix_ and max(ix_) - min(ix_) > len(pos) + 4:
                                    continue
                                if len(pos) > best:
                                    best, cands = len(pos), [(fn_, blk_, key_,
                                                             pos, ix_)]
                                elif len(pos) == best:
                                    cands.append((fn_, blk_, key_, pos, ix_))
                            uniq = {(c[0], c[1], c[2]) for c in cands}
                            if best < 4 or len(uniq) != 1:
                                break
                            fn_, blk_, key_, pos, ix_ = cands[0]
                            for a, b in zip(pos, ix_):
                                assigned[todo[a]] = ((fn_, blk_, key_), b)
                        if assigned:
                            break
                    if assigned:
                        break
                if not assigned:
                    miss += 1
                    continue
                # A cell is emitted only when EVERY figure in it is assigned
                # and all of them come from the SAME lab row - a cell spliced
                # from two sources would need a template this format cannot
                # express, and guessing one is how a correct figure gets
                # overwritten.  Those cells stay hand-maintained.
                srcmap = {k: v for k, v in assigned.items()}
                pool0 = {(a, b, c2): d for a, b, c2, d in pools[0][1]}
                for ci, c in enumerate(cells):
                    ks = [k for k in range(len(flat)) if owner[k] == ci]
                    if not ks or any(k not in srcmap for k in ks):
                        continue
                    srcs = {srcmap[k][0] for k in ks}
                    if len(srcs) != 1:
                        continue
                    src = srcs.pop()
                    mine = [srcmap[k][1] for k in ks]
                    tpl = NUM.sub("\u0000", c)
                    tpl = re.sub(r"(?:&minus;|-|\u2212)\u0000", "\u0000", tpl)
                    parts = tpl.split("\u0000")
                    tpl = "".join(p + ("{%d}" % i if i < len(parts) - 1 else "")
                                  for i, p in enumerate(parts))
                    # THE GATE: rendering this cell from the lab output the
                    # paper was built against must reproduce it exactly.  A
                    # coincidence of numbers written into the map would rewrite
                    # a correct figure on every future build, for ever.
                    toks0 = pool0.get(src)
                    if toks0 is None or max(mine) >= len(toks0):
                        rejected += 1
                        continue
                    try:
                        shown = tpl.format(*[_render(toks0[k]) for k in mine])
                    except Exception:
                        rejected += 1
                        continue
                    if shown.strip() != c.strip():
                        rejected += 1
                        continue
                    rows_out.append([pf, str(ti), str(ri), str(ci),
                                     src[0], src[1], src[2],
                                     ",".join(map(str, mine)), tpl])
    outp = os.path.join(HERE, "papers", "TABLE_MAP.tsv")
    with open(outp, "w", encoding="utf-8") as fh:
        fh.write("paper\ttable\trow\tcell\tlab\tblock\tkey\ttokens\ttemplate\n")
        for r in rows_out:
            fh.write("\t".join(r) + "\n")
    mapped = len({(r[0], r[1], r[2]) for r in rows_out})
    print(f"  paper rows considered : {seen}")
    print(f"  rows mapped           : {mapped}")
    print(f"  ambiguous (skipped)   : {amb}")
    print(f"  unmatched (skipped)   : {miss}")
    print(f"  REJECTED by the map gate (did not reproduce the "
          f"paper from its own lab output) : {rejected}")
    print(f"  cells mapped          : {len(rows_out)}")
    print(f"  -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
