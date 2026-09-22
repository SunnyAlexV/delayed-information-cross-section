# Appraisal-based / repeat-sales price series

Source: Federal Reserve Bank of St. Louis (FRED), `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>`
Retrieved: 2026-09-17.

Every file below was fetched at the FRED origin, a SHA-256 digest was computed
**at the source before transfer**, and the digest was re-computed on the stored
file after transfer. Both digests are recorded here so the transfer is verifiable
rather than asserted.

| file | FRED id | rows | bytes | SHA-256 (source == stored) |
|---|---|---|---|---|
| `CSUSHPISA_fred.csv`  | CSUSHPISA  | 474 | 8877 | `2db908a07c1470823db687d4a70e67ddd6e6961a0826e41a049e23c2631032d4` |
| `CSUSHPINSA_fred.csv` | CSUSHPINSA | 474 | 8878 | `d60c389a1b0596f1d034c6a19710e4dc2751bafdbeee7aac8bd3a14c6d886525` |
| `USSTHPI_fred.csv`    | USSTHPI    | 206 | 3713 | `c02d4187c8c9836d4049fe2005ab34129326e7538d2cd923173c2839ca25df16` |

Verify with:

    sha256sum data/illiquid/*.csv

## The metro panel

`CaseShiller_metro20_NSA.csv` -- 474 months (1987-01 to 2026-06) x 20
metropolitan Case-Shiller home price indices, **not** seasonally adjusted, wide
format on a shared date axis, 9143 observations, 83995 bytes,
SHA-256 `2a1399193a2966aba24c0a59da22c7a8b5c75e065871125f7c96121e62952ed7`.

Columns are the FRED ids: ATXRNSA (Atlanta), BOXRNSA (Boston), CRXRNSA
(Charlotte), CHXRNSA (Chicago), CEXRNSA (Cleveland), DAXRNSA (Dallas), DNXRNSA
(Denver), DEXRNSA (Detroit), LVXRNSA (Las Vegas), LXXRNSA (Los Angeles),
MIXRNSA (Miami), MNXRNSA (Minneapolis), NYXRNSA (New York), PHXRNSA (Phoenix),
POXRNSA (Portland), SDXRNSA (San Diego), SFXRNSA (San Francisco), SEXRNSA
(Seattle), TPXRNSA (Tampa), WDXRNSA (Washington DC).

Fourteen columns are complete from 1987-01. Six begin later and are blank
before their first published month: Minneapolis and Phoenix 1989-01, Seattle
1990-01, Atlanta and Detroit 1991-01, Dallas 2000-01. Detroit has no 2026-06
figure yet. Nothing is imputed; blanks are blanks.

### The one transformation applied, and why it is reversible

FRED serves these series with ten decimal places. Values here are **rounded to
four decimals**. The largest relative error this introduces anywhere in the
panel is 1.0e-6 -- roughly nine orders of magnitude below the sampling error of
a repeat-sales index -- and the step is exactly reproducible: download each
`<ID>` from `fredgraph.csv`, round each value to four decimals, drop trailing
zeros, join on the date axis, and the digest above is what you get. The
national series in the table at the top of this file are **not** rounded; they
are byte-identical to FRED.

### How the transfer was checked

The panel was assembled at the FRED origin and split into four blocks. A
SHA-256 digest was computed at the source for each block, for the header line,
and for the assembled file. Every block was verified on arrival before the file
was concatenated, and the assembled file's digest was checked against the
source's. All six digests matched. This is recorded because a data set moved by
any means deserves a check that fails loudly rather than a claim that it
arrived intact.

## What each series is, and why it is here

**CSUSHPISA** -- S&P Cotality (formerly CoreLogic) Case-Shiller U.S. National
Home Price Index, seasonally adjusted, monthly, 1987-01 to 2026-06.
Repeat-sales. Each monthly figure is a three-month moving average of closings
by construction, and closings themselves lag contract by weeks, so the mark is
smoothed and stale **before** any seasonal adjustment is applied. Published with
roughly a two-month lag. No option chain exists on it.

**CSUSHPINSA** -- the same index without seasonal adjustment. Seasonal
adjustment is a filter; on the published vintage it is two-sided and therefore
peeks. Any forecasting exercise in this repository uses the **NSA** series and
removes seasonality causally (month-of-year means estimated on the training
block only). The SA series is kept solely so that the sensitivity of a result to
that choice can be shown rather than argued.

**USSTHPI** -- FHFA all-transactions house price index, United States,
**quarterly**, 1975-Q1 to 2026-Q2. Longer history, coarser clock. It is here to
show where the frequency limit established in `lab54` actually bites: the
interval-width calculation there says a 40-point interval needs about 121
month-marks; at four marks a year that is thirty years of test sample, which the
series has, and a 30-point interval needs 215, which it does not.

## What is NOT claimed

These are *index* levels, not tradeable marks. Nobody can hold CSUSHPISA. The
series is used here as an instance of the object the main paper is about -- a
periodically published, smoothed, stale valuation of an illiquid asset -- and
not as a portfolio.
