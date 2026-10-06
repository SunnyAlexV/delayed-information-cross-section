"""
verify_data.py - check your copy of the inputs against the manifest.

    python verify_data.py <folder>      # default: ./data

WHAT THIS IS FOR
----------------
The repository cannot ship the data.  Every vendor behind it prohibits
redistribution (data/README.md sets out which terms, source by source), so what
ships is data/MANIFEST.tsv: for each input file, the SHA-256 of the exact bytes
the paper's numbers were computed from, and the SHA-256 of the derived series -
date and closing values, rounded to four decimals - taken over that file's own
dates.

Run this after obtaining the files yourself.  It tells you, per file, which of
three situations you are in:

    IDENTICAL   your bytes are our bytes.  Every number in the paper follows
                from a rerun, with nothing further to argue about.

    SAME DATA   the files differ but the derived series agrees on every date the
                two have in common, and the difference is confined to rows
                outside our span.  This is what a fresh export looks like: the
                vendor has added recent sessions since September 2026.  The
                paper's window is inside yours, so `run_all.py --check` should
                still reproduce the stored output.

    DIFFERENT   the two disagree on dates you both hold.  That is not a stale
                export, it is a different series - a revision, a different
                symbol, or a different column read as the close - and any
                mismatch in the results is explained by the data rather than by
                the code.  This is the case worth knowing about, and it is the
                one a digest over whole files cannot distinguish from the one
                above.

The third state is the reason this file exists.  A bare `sha256sum` says only
"different", which is useless advice to somebody who downloaded the same series
a month later.
"""

import csv
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "data", "MANIFEST.tsv")

sys.path.insert(0, HERE)
from build_manifest import derived  # noqa: E402


def main(folder=None):
    folder = os.path.abspath(folder or os.path.join(HERE, "data"))
    if not os.path.isfile(MANIFEST):
        raise SystemExit(f"missing {MANIFEST}")
    rows = list(csv.DictReader(open(MANIFEST), delimiter="\t"))
    print(f"manifest: {len(rows)} files")
    print(f"your folder: {folder}\n")
    state = {"IDENTICAL": [], "SAME DATA": [], "DIFFERENT": [], "MISSING": []}
    for r in rows:
        path = os.path.join(folder, r["path"])
        if not os.path.isfile(path):
            state["MISSING"].append(r["path"])
            print(f"  MISSING    {r['path']}   ({r['source']})")
            continue
        import hashlib
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if got == r["sha256"]:
            state["IDENTICAL"].append(r["path"])
            print(f"  IDENTICAL  {r['path']}")
            continue
        n, lo, hi, s_sha, years = derived(path)
        if s_sha == r["series_sha256"]:
            state["SAME DATA"].append(r["path"])
            print(f"  SAME DATA  {r['path']}   (bytes differ, series digest matches)")
            continue
        # the informative case: compare on the dates inside our span only
        ok, shared, _ = compare_on_shared(path, r)
        if ok:
            state["SAME DATA"].append(r["path"])
            print(f"  SAME DATA  {r['path']}   (agrees on {shared} rows inside our "
                  f"span; your span {lo}..{hi} against ours "
                  f"{r['first']}..{r['last']})")
        else:
            state["DIFFERENT"].append(r["path"])
            bad = disagreeing_years(years, r["year_sha256"])
            where = (f", in {', '.join(bad[:6])}" + (" and later" if len(bad) > 6 else "")
                     if bad else "")
            print(f"  DIFFERENT  {r['path']}   (disagrees inside our span{where})")
    print()
    for k in ("IDENTICAL", "SAME DATA", "DIFFERENT", "MISSING"):
        print(f"  {k:>10}: {len(state[k])}")
    if state["DIFFERENT"] or state["MISSING"]:
        print("\n  Anything above IDENTICAL or SAME DATA means run_all.py --check is")
        print("  not expected to match the stored output, and the difference is in")
        print("  the inputs rather than in the code.")
        return 1
    print("\n  Every file is either our bytes or the same series with a longer tail.")
    print("  `python run_all.py --check` should reproduce the stored output.")
    return 0


def disagreeing_years(mine, theirs):
    """Which calendar years differ, from the per-year digests in the manifest.

    Years present in only one of the two are not disagreements: a longer export
    has years we do not, and that is the SAME DATA case.  Only years both sides
    carry, with different digests, are reported.
    """
    def parse(v):
        out = {}
        for part in (v or "").split(","):
            if ":" in part:
                y, h = part.split(":", 1)
                out[y] = h
        return out
    a, b = parse(mine), parse(theirs)
    return sorted(y for y in set(a) & set(b) if a[y] != b[y])


def compare_on_shared(path, rec):
    """Do we agree on the dates we both hold?  Needs the reference series.

    The manifest stores a digest, not the series, so this can only be answered
    when the digest was taken over a span we still know: we compare the file
    against its own recorded span and report whether the rows inside that span
    reproduce the recorded digest.  A file whose extra rows are all OUTSIDE our
    span and whose inside-span rows hash correctly is the same data.
    """
    try:
        lo, hi = rec["first"], rec["last"]
        if not lo or not hi:
            return False, 0, None
        df = pd.read_csv(path)
    except Exception:                                          # noqa: BLE001
        return False, 0, None
    cols = {c.strip().lower(): c for c in df.columns}
    datecol = next((cols[k] for k in ("date", "observation_date") if k in cols), None)
    if datecol is None:
        return False, 0, None
    d = pd.to_datetime(df[datecol], errors="coerce", format="mixed")
    keep = df[(d >= lo) & (d <= hi)].copy()
    if keep.empty:
        return False, 0, None
    tmp = os.path.join(os.path.dirname(os.path.abspath(path)),
                       ".verify_data_window.csv")
    try:
        keep.to_csv(tmp, index=False)
        _, _, _, sha, _ = derived(tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return sha == rec["series_sha256"], len(keep), None


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
