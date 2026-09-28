"""Shared drawing style and helpers for the P2 thesis figures."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import re as _re
import matplotlib.text as _mtext

_PRETTY = [
    (r"\bDeltaW_l\b", "ΔW_l"), (r"\bDeltaW\b", "ΔW"), (r"\bA_hat_i\b", "Â_i"), (r"\blambda_f\b", "λ_f"), (r"\blambda_p\b", "λ_p"), (r"\blambda\b", "λ"),
    (r"\balpha_l\b", "α_l"), (r"\balpha\b", "α"), (r"\bOmega_l\b", "Ω_l"), (r"\bOmega\b", "Ω"),
    (r"\bomega_l\b", "ω_l"), (r"\bomega\b", "ω"), (r"\bdelta_l\b", "δ_l"), (r"\beps\b", "ε"),
    (r"\bsqrt\b ?", "√"), (r"\^2\b", "²"), (r"\bkappa\b", "κ"),
]


def pretty(text):
    """Render ASCII symbol names as Greek/Unicode for display (skips mathtext strings)."""
    if not isinstance(text, str) or "$" in text:
        return text
    for pat, rep in _PRETTY:
        text = _re.sub(pat, rep, text)
    return text


_orig_set_text = _mtext.Text.set_text


def _set_text(self, s):
    return _orig_set_text(self, pretty(s) if isinstance(s, str) else s)


_mtext.Text.set_text = _set_text

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "Thesis_Paper" / "figures"
TAB = ROOT / "Thesis_Paper" / "tables"
DATA = ROOT / "Thesis_Paper" / "data"

# Validated categorical slots (reference palette, light mode, white surface)
BLUE = "#2a78d6"  # Standard RL-LoRA
ORANGE = "#eb6834"  # HLoRA-RL (direct transfer)
AQUA = "#1baf7a"  # third series (always direct-labelled)
GRAY = "#898781"  # frozen reference / muted ink
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

# Status palette (always paired with a text label)
GOOD = "#0ca30c"
WARN = "#fab219"
CRIT = "#d03b3b"

# Diagram role fills (fill, edge)
FROZEN = ("#e8eef7", "#466489")
TRAIN = ("#ffecd2", "#a86215")
PERSIST = ("#efe8fa", "#7252a3")
EVID = ("#e5f3e9", "#36764b")
NEUTRAL = ("#f4f4f2", "#52514e")
ALERT = ("#fde8e6", "#b0413e")
BLUEBOX = ("#e3eefb", "#2a78d6")

SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 300,
        "mathtext.fontset": "dejavusans",
    }
)


def clean_axes(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def save(fig, name: str, folder: Path = FIG) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def canvas(width: float, height: float, xmax: float = 100, ymax: float = 100):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body=None, role=NEUTRAL, title_size=8.5, body_size=7.2,
        wrap=None, radius=1.2, bold=True, align="center", lw=1.1):
    """Rounded box with a bold title and optional wrapped body lines (data coords)."""
    fc, ec = role
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    )
    lines = []
    if body:
        items = body if isinstance(body, (list, tuple)) else [body]
        for item in items:
            if wrap:
                lines.extend(textwrap.wrap(item, wrap) or [""])
            else:
                lines.append(item)
    cx = x + w / 2 if align == "center" else x + 1.2
    ha = "center" if align == "center" else "left"
    if lines:
        ax.text(cx, y + h - 1.0, title, ha=ha, va="top", fontsize=title_size,
                fontweight="bold" if bold else "normal", color=INK, zorder=3)
        ax.text(cx, y + h - 1.0 - title_size * 0.42, "\n".join(lines), ha=ha, va="top",
                fontsize=body_size, color=INK2, zorder=3, linespacing=1.25)
    else:
        ax.text(cx, y + h / 2, title, ha=ha, va="center", fontsize=title_size,
                fontweight="bold" if bold else "normal", color=INK, zorder=3,
                linespacing=1.2)


def arrow(ax, start, end, color=INK2, lw=1.1, style="-|>", ls="-", rad=0.0, label=None,
          label_pos=0.5, label_offset=(0, 0.8), label_size=6.8, shrink=0.0, zorder=1):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=9, color=color,
                            lw=lw, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
                            shrinkA=shrink, shrinkB=shrink, zorder=zorder)
    ax.add_patch(patch)
    if label:
        lx = start[0] + (end[0] - start[0]) * label_pos + label_offset[0]
        ly = start[1] + (end[1] - start[1]) * label_pos + label_offset[1]
        ax.text(lx, ly, label, fontsize=label_size, color=INK2, ha="center", va="bottom",
                zorder=4, bbox=dict(fc=SURFACE, ec="none", pad=0.4))


def status_chip(ax, x, y, status: str, size=6.3):
    """Status label with a colored dot; never color alone."""
    color = {"Implemented": GOOD, "Partial": WARN, "Planned": GRAY, "Complete": GOOD}[status]
    ax.plot([x], [y], marker="o", ms=4.2, color=color, zorder=5, mec=SURFACE, mew=0.8)
    ax.text(x + 1.1, y, status, fontsize=size, va="center", ha="left", color=INK2, zorder=5)
