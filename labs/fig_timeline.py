"""
fig_timeline.py - draw the training and forecast timeline of Section 4.

    python fig_timeline.py            print the SVG to stdout
    python fig_timeline.py --check    print the constants it encodes and stop

WHY THIS FILE EXISTS
--------------------
Every other figure in this paper plots numbers a lab produced.  This one draws a
design, and a design is exactly the kind of thing that gets drawn wrong: a
diagram with the training window on the wrong side of the forecast origin, or
the label resolving before the cut, would misrepresent the experiment while
looking authoritative.  So the coordinates are computed from the same constants
the code uses rather than placed by hand, and the constants are imported from
lab05_robustness rather than retyped.  If the design changes, the figure moves.

WHAT THE DIAGRAM HAS TO SHOW, AND WHY EACH PIECE IS THERE
---------------------------------------------------------
Six dates matter and a reader currently has to assemble them from prose in
three different subsections:

    t                  the forecast origin: where the forecaster stands
    t - delta          the domestic mark they are allowed to see
    t (foreign)        the foreign closes, which are current, because those
                       exchanges shut before the target's does
    t + h              when the label resolves, h = 5 trading days later
    t - delta - h      where training stops, which is NOT t: the last label
                       that had resolved by the time the stale mark was current
    every REFIT days   when the coefficients are re-estimated

The cut at t - delta - h is the piece that most needs a picture.  It is a purge
in the sense the cross-validation literature means, it is load-bearing (Section
S7 measures what removing it inflates), and in prose it reads like an
implementation detail rather than the reason the design is honest.

ACCESSIBILITY
-------------
Colour carries nothing here that text does not.  Every band and marker is
labelled, the two hues are the pair already used in Figure 3 of this paper, and
the figure reads correctly in greyscale and to a colour-blind reader because the
labels, not the fills, identify the pieces.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab05_robustness as L  # noqa: E402

W, H = 720, 320
INK, MUTE, GRID = "#333", "#6B7280", "#E4E8EE"
BLUE, BROWN = "#1A5FA8", "#9A5210"
MONO = "DejaVu Sans Mono, monospace"

DELTA = 55          # the delay the paper leads with, in trading days
LEFT, RIGHT = 62, 704
LABEL_W = 124       # the left column that names each row, so nothing collides
GAP_W = 116         # reserved for the withheld strip in the upper panel


def fitting_bands():
    """Burn-in, training and validation, on a compressed but monotone scale.

    The three blocks are hundreds of days and the neighbourhood of t is tens,
    so one linear axis would render the interesting end as a single pixel.  The
    upper panel is therefore compressed and says so; the lower panel is linear
    and is where the days are read.
    """
    total = L.MED + L.TRAIN + L.VAL
    span = RIGHT - GAP_W - LEFT
    x, out = LEFT, []
    for key, n in (("burn", L.MED), ("train", L.TRAIN), ("val", L.VAL)):
        w = span * n / total
        out.append((key, n, x, w))
        x += w
    return out, x


def svg():
    p = []
    a = p.append
    a(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Two-panel timeline of '
      'the walk-forward design. The upper panel shows a burn-in block of 252 '
      'trading days, a training block of 1250 and a validation tail of 250, '
      'ending at the training cut, followed by a withheld gap of delta plus h '
      'days before the forecast origin. The lower panel shows four rows on one '
      'linear axis: the training window stopping delta plus h days before t, '
      "the target\u2019s own data stopping delta days before t and stale "
      'thereafter, the foreign closes running right up to t, and the label '
      'resolving h days after t.">')

    # ================= upper panel: the fitting window ==================
    bands, xcut = fitting_bands()
    y0, bh = 66, 28
    a(f'<text x="{LEFT}" y="24" fill="{INK}" font-family="{MONO}" '
      f'font-size="11" font-weight="bold">What the model is fitted on, and '
      'what is held back</text>')
    a(f'<text x="{LEFT}" y="40" fill="{MUTE}" font-family="{MONO}" '
      f'font-size="9">re-estimated every {L.REFIT} trading days; this panel is '
      'compressed, the one below is to scale</text>')

    fills = {"burn": "#F4F6F9", "train": "#DCE6F3", "val": "#EFE3D6"}
    for key, n, x, w in bands:
        a(f'<rect x="{x:.1f}" y="{y0}" width="{w:.1f}" height="{bh}" '
          f'fill="{fills[key]}" stroke="{GRID}"/>')
        a(f'<text x="{x + w / 2:.1f}" y="{y0 + 18}" fill="{INK}" '
          f'font-family="{MONO}" font-size="9.5" text-anchor="middle">'
          f'{n}d</text>')

    a(f'<rect x="{xcut:.1f}" y="{y0}" width="{RIGHT - xcut:.1f}" height="{bh}" '
      f'fill="none" stroke="{BROWN}" stroke-width="1.1" '
      'stroke-dasharray="3 3"/>')
    a(f'<text x="{(xcut + RIGHT) / 2:.1f}" y="{y0 + 18}" fill="{BROWN}" '
      f'font-family="{MONO}" font-size="9.5" text-anchor="middle">'
      f'{DELTA + L.HORIZON}d</text>')
    a(f'<line x1="{xcut:.1f}" y1="{y0 - 5}" x2="{xcut:.1f}" y2="{y0 + bh + 2}" '
      f'stroke="{BROWN}" stroke-width="1.6"/>')
    a(f'<text x="{xcut:.1f}" y="{y0 - 10}" fill="{BROWN}" font-family="{MONO}" '
      f'font-size="8.8" text-anchor="middle">the cut</text>')
    a(f'<line x1="{RIGHT:.1f}" y1="{y0 - 5}" x2="{RIGHT:.1f}" y2="{y0 + bh + 2}" '
      f'stroke="{BLUE}" stroke-width="1.6"/>')
    a(f'<text x="{RIGHT:.1f}" y="{y0 - 10}" fill="{BLUE}" font-family="{MONO}" '
      f'font-size="8.8" text-anchor="end">t</text>')

    # One caption line for the whole strip, so no label has to fit inside a
    # band narrower than its own text.  The bands carry day counts only.
    a(f'<text x="{LEFT}" y="{y0 + bh + 15}" fill="{MUTE}" font-family="{MONO}" '
      f'font-size="8.4">left to right: burn-in for the trailing median, '
      'training for the coefficients, validation for the ridge penalty, then '
      f'&#948;+h withheld</text>')

    # ================= lower panel: one linear axis ======================
    x0, x1 = LEFT + LABEL_W, RIGHT
    lo, hi = -(DELTA + L.HORIZON) - 10, L.HORIZON + 30
    def X(d):
        return x0 + (x1 - x0) * (d - lo) / (hi - lo)

    yh = 158
    a(f'<text x="{LEFT}" y="{yh - 24}" fill="{INK}" font-family="{MONO}" '
      f'font-size="11" font-weight="bold">What is available at one forecast '
      f'origin, at &#948; = {DELTA}</text>')

    a(f'<line x1="{X(0):.1f}" y1="{yh - 10}" x2="{X(0):.1f}" y2="{yh + 118}" '
      f'stroke="{BLUE}" stroke-width="1.4"/>')
    a(f'<text x="{X(0):.1f}" y="{yh - 13}" fill="{BLUE}" font-family="{MONO}" '
      f'font-size="9.5" text-anchor="middle">t</text>')
    xc = X(-(DELTA + L.HORIZON))
    a(f'<line x1="{xc:.1f}" y1="{yh - 4}" x2="{xc:.1f}" y2="{yh + 118}" '
      f'stroke="{BROWN}" stroke-width="1.1" stroke-dasharray="2 3"/>')

    # (row name, bar ends at, gap colour, text placed in the gap)
    rows = [
        ("training window", -(DELTA + L.HORIZON), BROWN,
         "held back: the last label that had resolved"),
        ("the target's own data", -DELTA, BROWN,
         f"stale for &#948; = {DELTA} days"),
    ]
    y, rh, step = yh, 18, 28
    for name, end, col, inner in rows:
        a(f'<text x="{x0 - 10}" y="{y + 13}" fill="{INK}" '
          f'font-family="{MONO}" font-size="9.2" text-anchor="end">{name}</text>')
        a(f'<rect x="{x0:.1f}" y="{y}" width="{X(end) - x0:.1f}" height="{rh}" '
          f'fill="#DCE6F3" stroke="{GRID}"/>')
        a(f'<rect x="{X(end):.1f}" y="{y}" width="{X(0) - X(end):.1f}" '
          f'height="{rh}" fill="none" stroke="{col}" stroke-width="1" '
          'stroke-dasharray="2 3"/>')
        a(f'<text x="{(X(end) + X(0)) / 2:.1f}" y="{y + 13}" fill="{col}" '
          f'font-family="{MONO}" font-size="8.4" text-anchor="middle">'
          f'{inner}</text>')
        y += step

    a(f'<text x="{x0 - 10}" y="{y + 13}" fill="{INK}" font-family="{MONO}" '
      f'font-size="9.2" text-anchor="end">the foreign closes</text>')
    a(f'<rect x="{x0:.1f}" y="{y}" width="{X(0) - x0:.1f}" height="{rh}" '
      f'fill="#DCE6F3" stroke="{GRID}"/>')
    a(f'<text x="{X(0) - 8:.1f}" y="{y + 13}" fill="{BLUE}" '
      f'font-family="{MONO}" font-size="8.4" text-anchor="end">current at t: '
      "those exchanges shut before the target\u2019s</text>")
    y += step

    a(f'<text x="{x0 - 10}" y="{y + 13}" fill="{INK}" font-family="{MONO}" '
      f'font-size="9.2" text-anchor="end">the label</text>')
    a(f'<rect x="{X(0):.1f}" y="{y}" width="{X(L.HORIZON) - X(0):.1f}" '
      f'height="{rh}" fill="#EFE3D6" stroke="{GRID}"/>')
    a(f'<text x="{X(L.HORIZON) + 8:.1f}" y="{y + 13}" fill="{INK}" '
      f'font-family="{MONO}" font-size="8.4">resolves at t+h, h = '
      f'{L.HORIZON}d</text>')
    a(f'<text x="{X(0) - 8:.1f}" y="{y + 13}" fill="{MUTE}" '
      f'font-family="{MONO}" font-size="8.4" text-anchor="end">unknown at the '
      'origin, by construction</text>')

    xa, xb = X(-DELTA), X(0)
    ybk = yh + 122
    a(f'<path d="M {xa:.1f} {ybk} L {xa:.1f} {ybk + 6} L {xb:.1f} {ybk + 6} '
      f'L {xb:.1f} {ybk}" fill="none" stroke="{BLUE}" stroke-width="1.2"/>')
    a(f'<text x="{(xa + xb) / 2:.1f}" y="{ybk + 20}" fill="{BLUE}" '
      f'font-family="{MONO}" font-size="9.2" text-anchor="middle">'
      f'&#948;, the quantity this paper sweeps from 0 to {DELTA}</text>')
    a('</svg>')
    return "\n".join(x for x in p if x)


def main(argv):
    if "--check" in argv:
        print(f"  MED (burn-in)        {L.MED}")
        print(f"  TRAIN                {L.TRAIN}")
        print(f"  VAL                  {L.VAL}")
        print(f"  REFIT                {L.REFIT}")
        print(f"  HORIZON h            {L.HORIZON}")
        print(f"  delta drawn          {DELTA}")
        print(f"  training cut at      t - {DELTA + L.HORIZON}")
        return
    print(svg())


if __name__ == "__main__":
    main(sys.argv)
