"""
build_manifest.py - write data/MANIFEST.tsv from a local data folder.

    python build_manifest.py <folder>       # default: ./data

WHY THIS FILE EXISTS
--------------------
Every vendor behind this project's inputs prohibits redistributing their files.
Investing.com's terms forbid distributing data from the site without written
permission; Cboe's permit one downloaded copy for personal non-commercial use
and nothing further; Yahoo's forbid reproducing or distributing content; and the
S&P Cotality Case-Shiller series carry FRED's "Copyrighted: Pre-approval
Required" label, which allows non-commercial educational or personal use only.
Of everything this project reads, exactly one file is redistributable: the FHFA
index, which FRED labels "Public Domain: Citation Requested".

So the repository cannot ship the data, and the reproducibility claim has to be
made a different way: record, for every input, enough to prove that a reader who
obtains the same file is feeding the scripts the same bytes we fed them.  This
file writes that record.  It is run against a private working copy and its
output, data/MANIFEST.tsv, is what ships.

WHAT IS RECORDED, AND WHY EACH COLUMN EARNS ITS PLACE
-----------------------------------------------------
    sha256        the exact bytes.  A reader holding our file matches here and
                  needs nothing else.
    bytes, rows   catches a truncated or partially written download, which is
                  the failure a digest reports as "different file" without
                  saying how.
    first, last   the date span, so a reader with a FRESH export - which will
                  have more recent rows than ours and cannot match the digest -
                  can see immediately whether the difference is only the tail.
    series_sha    the digest of the DERIVED series on the dates we both hold:
                  date and close, rounded, in a fixed text form.  This is the
                  column that makes a fresh export checkable.  If a reader's
                  export agrees with ours on every shared date, this matches on
                  the intersection even though the files do not.

    year_sha      the same digest taken one calendar year at a time, eight hex
                  characters each.  It publishes no value and no date, and it
                  turns "your file disagrees somewhere" into "your file
                  disagrees in 2008", which is a difference somebody can act on.

The derived columns are the point of the exercise.  A digest of a vendor's CSV is a
statement about that vendor's formatting; a digest of the closing prices we
actually used is a statement about the input to the experiment.
"""

import hashlib
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "labs"))

# Where each file comes from, and whether it may be redistributed.  The licence
# column is a statement about the vendor's terms as they read in September 2026,
# not legal advice, and the terms are linked in data/README.md.
SOURCES = [
    ("SPX_*.csv", "Investing.com", "no"),
    ("N225_*.csv", "Investing.com", "no"),
    ("AXJO_*.csv", "Investing.com", "no"),
    ("HSI_*.csv", "Investing.com", "no"),
    ("NSEI_*.csv", "Investing.com", "no"),
    ("FTSE_*.csv", "Investing.com", "no"),
    ("DAX_*.csv", "Investing.com", "no"),
    ("BVSP_*.csv", "Investing.com", "no"),
    ("DAX_New_Volatility_*.csv", "Investing.com", "no"),
    ("CBOE_Volatility_Index_*.csv", "Investing.com", "no"),
    ("STOXX_50_Volatility_*.csv", "Investing.com", "no"),
    ("VSTOXX_Mini_Futures_*.csv", "Investing.com", "no"),
    ("VIX_History.csv", "Cboe", "no"),
    ("VIX9D_History.csv", "Cboe", "no"),
    ("single_market/XYZ.csv", "Investing.com", "no"),
    ("frontier/*.csv", "Yahoo Finance", "no"),
    ("illiquid/CaseShiller_metro20_NSA.csv", "FRED / S&P Cotality", "no"),
    ("illiquid/CSUSHPISA_fred.csv", "FRED / S&P Cotality", "no"),
    ("illiquid/CSUSHPINSA_fred.csv", "FRED / S&P Cotality", "no"),
    ("illiquid/USSTHPI_fred.csv", "FRED / FHFA", "yes"),
]


def owner(rel):
    """Vendor and redistribution flag for one relative path."""
    import fnmatch
    for pat, src, redist in SOURCES:
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(os.path.basename(rel), pat):
            return src, redist
    return "unrecorded", "unknown"


def derived(path):
    """(rows, first, last, digest of date+close) for whichever format this is.

    Three formats appear in this project and all three are handled here rather
    than in three places: the retail export (Date, Price), Cboe's own history
    (DATE, CLOSE), and a FRED download (observation_date plus one or more
    series columns).  The digest covers every numeric column, so the metro
    panel's twenty series are all inside it.
    """
    try:
        df = pd.read_csv(path)
    except Exception as exc:                                  # noqa: BLE001
        return None, None, None, f"unreadable: {type(exc).__name__}", ""
    cols = {c.strip().lower(): c for c in df.columns}
    datecol = next((cols[k] for k in ("date", "observation_date") if k in cols), None)
    if datecol is None:
        return len(df), None, None, "no date column", ""
    d = pd.to_datetime(df[datecol], errors="coerce", format="mixed", dayfirst=False)
    num = []
    for c in df.columns:
        if c == datecol:
            continue
        v = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False),
                          errors="coerce")
        if v.notna().sum() > 0.5 * len(v):
            num.append((c, v))
    if not num:
        return len(df), None, None, "no numeric column", ""
    keep = pd.DataFrame({"d": d.dt.strftime("%Y-%m-%d")})
    for c, v in num:
        keep[c] = v.round(4)
    keep = keep.dropna(subset=["d"]).sort_values("d")
    text = "\n".join(
        "\t".join([r.d] + [f"{getattr(r, c):.4f}" if pd.notna(getattr(r, c)) else ""
                           for c, _ in num])
        for r in keep.itertuples())
    per_year = {}
    for line in text.split("\n"):
        per_year.setdefault(line[:4], []).append(line)
    years = ",".join(
        f"{y}:{hashlib.sha256(chr(10).join(v).encode()).hexdigest()[:8]}"
        for y, v in sorted(per_year.items()))
    return (len(df), keep["d"].iloc[0], keep["d"].iloc[-1],
            hashlib.sha256(text.encode()).hexdigest(), years)


def conventions():
    """Write data/CONVENTIONS.tsv: the timing rules, from the code that uses them.

    A manifest of digests says the bytes are right.  It says nothing about how
    those bytes were interpreted, and the interpretation is where a replication
    silently diverges: which close time a series is stamped with, whether a
    market contributes today's session or yesterday's, and what happens on a
    day one exchange is shut.  Those rules live in lab05_robustness and were
    described only in prose, so they are written out here in a form a machine
    can read and a reader can check.
    """
    import lab05_robustness as L
    tgt = "SPX"
    rows = []
    for tag, close in sorted(L.CLOSE_UTC.items(), key=lambda kv: (kv[1], kv[0])):
        admissible = close < L.CLOSE_UTC[tgt]
        rows.append(dict(
            series=tag, close_utc=f"{int(close):02d}:{int(round((close % 1) * 60)):02d}",
            role="target" if tag == tgt else "foreign peer",
            contributes=("same session" if admissible else
                         "previous session" if tag != tgt else "n/a"),
            rule=("closes before the target, so today's close is admissible"
                  if admissible else
                  "closes at or after the target, so the previous session is used"
                  if tag != tgt else "the target itself")))
    # the implied-volatility series, whose admissibility is the awkward case
    rows.append(dict(series="VDAX", close_utc="16:30", role="implied volatility",
                     contributes="same session",
                     rule="Frankfurt closes before New York, so admissible outright"))
    rows.append(dict(series="VIX", close_utc="21:15", role="implied volatility",
                     contributes="previous session",
                     rule="printed close is stamped 15 minutes AFTER the target's, "
                          "so the lagged value leads and the same-day value is a "
                          "sensitivity"))
    rows.append(dict(series="VIX9D", close_utc="21:15", role="implied volatility",
                     contributes="previous session",
                     rule="as VIX; used only in the matched-horizon race"))
    rows.append(dict(series="Case-Shiller metros", close_utc="n/a",
                     role="monthly panel", contributes="same month",
                     rule="all twenty publish on the same day with the same "
                          "two-month lag, so delta is RELATIVE staleness and not "
                          "that lag"))
    cols = ["series", "close_utc", "role", "contributes", "rule"]
    out = os.path.join(HERE, "data", "CONVENTIONS.tsv")
    with open(out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        fh.write("# daylight saving is ignored: it moves a close by an hour and "
                 "never reorders the blocks\n")
        fh.write("# a day any series is missing is dropped from that model's rows "
                 "rather than imputed\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"wrote {out}: {len(rows)} series")


def main(folder=None):
    folder = os.path.abspath(folder or os.path.join(HERE, "data"))
    rows = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if not f.lower().endswith(".csv"):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, folder).replace(os.sep, "/")
            src, redist = owner(rel)
            n, lo, hi, sha_series, years = derived(path)
            rows.append(dict(
                path=rel, source=src, redistributable=redist,
                bytes=os.path.getsize(path),
                sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
                rows=n, first=lo or "", last=hi or "", series_sha256=sha_series,
                year_sha256=years))
    rows.sort(key=lambda r: r["path"])
    cols = ["path", "source", "redistributable", "bytes", "rows", "first",
            "last", "sha256", "series_sha256", "year_sha256"]
    out = os.path.join(HERE, "data", "MANIFEST.tsv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"wrote {out}: {len(rows)} files")
    bad = [r["path"] for r in rows if r["source"] == "unrecorded"]
    if bad:
        raise SystemExit("files with no recorded source: " + ", ".join(bad))
    n_yes = sum(1 for r in rows if r["redistributable"] == "yes")
    print(f"  {n_yes} of {len(rows)} may be redistributed; the rest may not.")
    conventions()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
