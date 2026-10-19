"""Figure style and layout checks for the manuscript figures.

All figures share one canvas width (16 cm, the text width of the manuscript),
Times-compatible STIX fonts, the Okabe-Ito palette and a legend placed below
the axes. save() refuses to write a figure in which a legend or any text sits
inside the plotting region, two text elements overlap, something is clipped by
the canvas, or plain (non-math) text contains '_' or '^'.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CM = 1.0 / 2.54
WIDTH_CM = 16.0
DPI = 600

# Okabe-Ito colors, safe under the common forms of color-vision deficiency
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
MARKERS = ("o", "s", "^", "D", "v", "P")
LINESTYLES = ("-", "--", "-.", (0, (5, 1.5)), (0, (1.5, 1.5)))
GREY = "0.35"

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.2,
    "ytick.minor.size": 2.2,
    "xtick.major.pad": 5.0,
    "ytick.major.pad": 5.0,
    "axes.grid": False,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.06,
    "figure.constrained_layout.w_pad": 0.06,
    "pdf.fonttype": 42,
    "savefig.dpi": DPI,
})


def style(i):
    """Color, marker and line style for series i."""
    return {"color": COLORS[i % len(COLORS)], "marker": MARKERS[i % len(MARKERS)],
            "linestyle": LINESTYLES[i % len(LINESTYLES)]}


def new_figure(height_cm=10.0):
    fig, ax = plt.subplots(figsize=(WIDTH_CM * CM, height_cm * CM))
    return fig, ax


def legend_below(fig, handles, ncol=2):
    """Figure legend under the axes, outside the plotting region.

    The number of columns is reduced until the legend fits the canvas width.
    """
    for n in range(ncol, 0, -1):
        leg = fig.legend(handles=handles, loc="outside lower center", ncol=n,
                         handlelength=2.6, columnspacing=1.8, handletextpad=0.6,
                         borderaxespad=0.2)
        fig.canvas.draw()
        if leg.get_window_extent().width <= 0.98 * fig.bbox.width or n == 1:
            return leg
        leg.remove()


def line_handle(label, **kw):
    return Line2D([], [], label=label, **kw)


# ---------------------------------------------------------------------------
# Layout checks
# ---------------------------------------------------------------------------
_MATH = re.compile(r"\$[^$]*\$")


def _overlap(a, b, tol=1.0):
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w > tol and h > tol


def _tick_labels(axis):
    """Tick labels that are actually drawn (tick location inside the view)."""
    lo, hi = sorted(axis.get_view_interval())
    pad = 1e-9 * max(abs(hi - lo), 1e-300)
    out = []
    for tick in axis.get_major_ticks() + axis.get_minor_ticks():
        loc = tick.get_loc()
        if loc is None or not (lo - pad <= loc <= hi + pad):
            continue
        for lab in (tick.label1, tick.label2):
            if lab.get_visible() and lab.get_text().strip():
                out.append(lab)
    return out


def layout_problems(fig):
    """List of layout defects; an empty list means the figure passes."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    problems, boxes = [], []          # boxes: (name, group, bbox)

    for j, leg in enumerate(fig.legends):
        boxes.append((f"legend {j}", f"legend {j}", leg.get_window_extent(r)))
    plots = [ax.get_window_extent(r) for ax in fig.axes]
    for i, ax in enumerate(fig.axes):
        if ax.get_legend() is not None:
            problems.append(f"axes {i}: legend placed on the axes")
        for t in ax.texts:
            if t.get_text().strip() and _overlap(t.get_window_extent(r), plots[i]):
                problems.append(f"axes {i}: text '{t.get_text()[:30]}' inside the plot")
        for name, t in (("title", ax.title), ("x label", ax.xaxis.label),
                        ("y label", ax.yaxis.label)):
            if t.get_text().strip():
                boxes.append((f"axes {i} {name}", f"axes {i} {name}",
                              t.get_window_extent(r)))
        for axis in (ax.xaxis, ax.yaxis):
            labels = _tick_labels(axis)
            for t in labels:
                boxes.append((f"axes {i} {axis.axis_name} tick '{t.get_text()}'",
                              f"axes {i} {axis.axis_name} ticks",
                              t.get_window_extent(r)))
            # neighboring labels on the same axis must not touch either
            ext = sorted((t.get_window_extent(r) for t in labels),
                         key=lambda b: (b.x0, b.y0))
            for a, b in zip(ext, ext[1:]):
                if _overlap(a, b, tol=0.0):
                    problems.append(f"axes {i}: {axis.axis_name} tick labels collide")

    for name, group, b in boxes:
        if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1:
            problems.append(f"{name} is clipped by the canvas")
        for i, plot in enumerate(plots):
            if not group.endswith("ticks") and _overlap(b, plot):
                problems.append(f"{name} overlaps the plot of axes {i}")
    for k, (na, ga, a) in enumerate(boxes):
        for nb, gb, b in boxes[k + 1:]:
            if ga != gb and _overlap(a, b):
                problems.append(f"{na} overlaps {nb}")

    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible():
            continue
        plain = _MATH.sub("", t.get_text())
        if "_" in plain or "^" in plain:
            problems.append(f"unformatted symbol in text '{t.get_text()[:40]}'")
    return sorted(set(problems))


def save(fig, stem):
    """Check the layout, then write PDF and 600 dpi PNG versions of the figure."""
    problems = layout_problems(fig)
    if problems:
        raise RuntimeError("figure layout check failed for " + str(stem) + ":\n  "
                           + "\n  ".join(problems))
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"),
                metadata={"Creator": None, "Producer": None, "CreationDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=DPI, metadata={"Software": None})
    return stem
