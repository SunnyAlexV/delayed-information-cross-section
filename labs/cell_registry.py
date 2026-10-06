"""cell_registry.py - the registry of reported cells, with their SERIES.

WHY THIS FILE EXISTS

lab62 audits the secondary grids by regex-scraping z-statistics out of the other
labs' printed output.  Scalars are all it has ever had, which is why its null is
"0.05n": you cannot resample a number.  An omnibus test of the pooled count needs
each cell's per-period series, its calendar group, its bandwidth and its critical
value, so that blocks can be drawn jointly across cells that share an evaluation
period and the cross-cell dependence survives into the null.

Nothing in the paper assembles that.  S37 says so in its own words - "no
family-wise control was applied across them" - and the deeper reason is that the
family was never written down.  You cannot correct across a family you have not
enumerated.

This module is the enumeration.  A lab calls record() once per reported cell, at
the point where the cell's loss differential already exists, and the registry
accumulates on disk.  No lab's own output changes; recording is a side effect.

USAGE, inside a lab, where the differential is already in hand:

    import cell_registry as REG
    REG.record(family="seven foreign closes given implied volatility",
               cell_id=f"lab51:{target}:d{delta}",
               target=target, delta=delta, estimand="GW increment",
               calendar_group=f"daily-{target}",
               index=dates, series=dl, bandwidth=lag, critical_value=1.96,
               provenance="post hoc")

Set REGISTRY_OFF=1 in the environment to disable recording entirely, which is
what the reproduction run does, so the stored output stays byte-identical.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("CELL_REGISTRY_DIR",
                     os.path.join(os.path.dirname(HERE), "cell_registry"))
OFF = os.environ.get("REGISTRY_OFF", "") == "1"


def _safe(s):
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))


def record(*, family, cell_id, series, index=None, calendar_group,
           bandwidth, critical_value, target=None, delta=None,
           estimand=None, provenance="post hoc", enters="exploratory-count"):
    """Append one cell to the registry.  Returns the path written, or None."""
    if OFF:
        return None
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 50:
        return None
    os.makedirs(OUT, exist_ok=True)
    stem = _safe(cell_id)
    np.save(os.path.join(OUT, stem + ".npy"), s)
    meta = dict(cell_id=cell_id, family=family, target=target, delta=delta,
                estimand=estimand, calendar_group=calendar_group,
                n=int(s.size), bandwidth=int(bandwidth),
                critical_value=float(critical_value),
                provenance=provenance, enters=enters,
                index_start=str(index[0]) if index is not None and len(index) else None,
                index_end=str(index[-1]) if index is not None and len(index) else None)
    with open(os.path.join(OUT, stem + ".json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return stem


def load_all():
    """Every recorded cell, as (meta, series) pairs."""
    if not os.path.isdir(OUT):
        return []
    out = []
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(OUT, f), encoding="utf-8") as fh:
            meta = json.load(fh)
        p = os.path.join(OUT, f[:-5] + ".npy")
        if os.path.isfile(p):
            out.append((meta, np.load(p)))
    return out
