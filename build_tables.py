"""build_tables.py - rewrite every mapped result-table cell from the labs.

    python build_tables.py            fill the tables, report what moved
    python build_tables.py --check    change nothing; exit 1 if anything would

WHY THIS FILE EXISTS
--------------------
The result tables were maintained by hand.  Each time a lab moved, roughly a
hundred cells went stale at once, verify_paper reported a hundred failures, and
they were patched one at a time - work that had to be redone on the next change,
and had been redone several times.

Nothing else in this project is maintained that way.  CONVENTIONS.tsv is written
from lab05's own constants so it cannot disagree with the code that applies
them.  build_variants fills figures from lab output and FAILS if a result figure
is typed into generated prose.  The result tables simply predate that machinery;
this file brings them into it.

papers/TABLE_MAP.tsv says, for every mapped cell, which lab file, which block of
it, which row key and which token that cell is a copy of, plus the template that
renders it in the paper's own notation.  derive_table_map.py builds that map.

WHAT IT REFUSES TO DO
---------------------
A map entry whose lab row has VANISHED is a hard failure, not a skip.  The map
is derived from the labs' own shape, so a row disappearing means the shape moved
and the map is stale - and a filler that quietly skipped it would leave a cell
frozen at whatever it last held while reporting success, which is the exact
failure mode this file exists to end.

--check is what belongs in the build: it asserts the tables already agree with
the labs rather than making them agree, so the stored papers remain the thing
being verified instead of something the verifier just rewrote.

WHAT IT DOES NOT COVER
----------------------
Only the rows derive_table_map.py could key unambiguously - it reports the
count, and the rest stay hand-maintained.  Those remain verify_paper's problem,
which is the right division: this file removes a class of work, it does not
remove the checking.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAP = os.path.join(HERE, "papers", "TABLE_MAP.tsv")
EXP = os.path.join(HERE, "expected_output")
NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")
CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
TR = re.compile(r"<tr[^>]*>.*?</tr>", re.S | re.I)
TABLE = re.compile(r"<table\b.*?</table>", re.S | re.I)


def lab_index():
    """(file, block, key) -> tokens, including the joined pairs the map uses."""
    idx = {}
    seen_keys = {}
    for fn in sorted(os.listdir(EXP)):
        if not fn.endswith(".txt"):
            continue
        blk = 0
        for ln in open(os.path.join(EXP, fn), encoding="utf-8",
                       errors="replace"):
            st = ln.strip()
            if st and set(st) <= set("=-") and len(st) > 20:
                blk += 1
                continue
            toks = ln.split()
            if len(toks) >= 3 and len(NUM.findall(ln)) >= 2:
                # must address rows exactly as derive_table_map.py does: a
                # block may hold several rows with the same leading label
                nth = seen_keys.get((fn, str(blk), toks[0]), 0)
                seen_keys[(fn, str(blk), toks[0])] = nth + 1
                idx[(fn, str(blk), f"{toks[0]}#{nth}")] = list(toks)
    for (fn, blk, key), toks in list(idx.items()):
        for (fn2, blk2, key2), toks2 in list(idx.items()):
            if fn2 == fn and key2 == key and blk2 != blk and "+" not in blk \
                    and "+" not in blk2:
                idx.setdefault((fn, f"{blk}+{blk2}", key), toks + toks2[1:])
    return idx


def render(tok):
    """A lab token in the paper's notation: a minus sign is an entity."""
    # The TEMPLATE carries the paper's own notation - its brackets, its per
    # cent sign, its thousands separator - so the token must arrive bare or the
    # two compose into '29.7%%'.  That doubling silently failed the map gate on
    # every percentage cell in the paper, which is most of the rate columns.
    tok = tok.strip("[],%").replace(",", "")
    if tok.startswith("-"):
        return "&minus;" + tok[1:]
    return tok


def main(argv):
    check = "--check" in argv
    rows = [l.rstrip("\n").split("\t")
            for l in open(MAP, encoding="utf-8")][1:]
    idx = lab_index()
    by_paper = {}
    for r in rows:
        by_paper.setdefault(r[0], []).append(r)

    changed = fails = 0
    for pf, entries in sorted(by_paper.items()):
        path = os.path.join(HERE, "papers", "source", pf)
        if not os.path.isfile(path):
            print(f"  FAIL  {pf} is in the map but not in papers/source")
            fails += 1
            continue
        t = open(path, encoding="utf-8").read()
        tabs = list(TABLE.finditer(t))
        edits = []
        for pf_, ti, ri, ci, lab, blk, key, toks, tpl in entries:
            ti, ri, ci = int(ti), int(ri), int(ci)
            want = idx.get((lab, blk, key))
            if want is None:
                print(f"  FAIL  {lab} block {blk} row '{key}' has vanished; "
                      f"the map is stale - re-run derive_table_map.py")
                fails += 1
                continue
            if ti >= len(tabs):
                print(f"  FAIL  {pf}: table {ti} no longer exists")
                fails += 1
                continue
            blkm = tabs[ti]
            trs = list(TR.finditer(blkm.group(0)))
            if ri >= len(trs):
                print(f"  FAIL  {pf}: table {ti} row {ri} no longer exists")
                fails += 1
                continue
            trtext = trs[ri].group(0)
            cells = CELL.findall(trtext)
            if ci >= len(cells):
                print(f"  FAIL  {pf}: table {ti} row {ri} cell {ci} is gone")
                fails += 1
                continue
            ks = [int(k) for k in toks.split(",")]
            if max(ks) >= len(want):
                print(f"  FAIL  {lab} block {blk} row '{key}' now has "
                      f"{len(want)} tokens; the map wants index {max(ks)}. "
                      f"That row's shape changed - re-run derive_table_map.py")
                fails += 1
                continue
            try:
                new = tpl.format(*[render(want[k]) for k in ks])
            except (IndexError, KeyError) as exc:
                print(f"  FAIL  {pf} table {ti} row {ri} cell {ci}: "
                      f"template {tpl!r} does not fit its tokens ({exc})")
                fails += 1
                continue
            if new != cells[ci]:
                edits.append((ti, ri, ci, cells[ci], new))
        if not edits:
            continue
        changed += len(edits)
        if check:
            for ti, ri, ci, old, new in edits[:6]:
                print(f"  STALE {pf} table {ti} row {ri} cell {ci}: "
                      f"{old.strip()!r} should be {new.strip()!r}")
            if len(edits) > 6:
                print(f"        ... and {len(edits) - 6} more in {pf}")
            continue
        # apply, deepest offset first so earlier edits do not move later ones
        for ti, ri, ci, old, new in sorted(edits, reverse=True):
            blkm = list(TABLE.finditer(t))[ti]
            bs, be = blkm.span()
            blktext = blkm.group(0)
            trm = list(TR.finditer(blktext))[ri]
            ts, te = trm.span()
            trtext = trm.group(0)
            cm = list(CELL.finditer(trtext))[ci]
            cs, ce = cm.span(1)
            trnew = trtext[:cs] + new + trtext[ce:]
            blknew = blktext[:ts] + trnew + blktext[te:]
            t = t[:bs] + blknew + t[be:]
        open(path, "w", encoding="utf-8").write(t)
        print(f"  {pf}: {len(edits)} cell(s) rewritten from the labs")

    n = len(rows)
    if check:
        print(f"build_tables --check: {n} mapped cells, {changed} stale, "
              f"{fails} broken map entries")
        return 1 if (changed or fails) else 0
    print(f"build_tables: {n} mapped cells, {changed} rewritten, "
          f"{fails} broken map entries")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
