"""Shared pieces of the paper assets: figure style, family styles, LaTeX tables and numbers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from maintadapt.plotting import (
    BASELINE,
    INK,
    INK_SECONDARY,
    MUTED,
    SERIES,
    apply_style,
)
from maintadapt.research.models.spaces import FAMILY_NAMES

# journal column and page widths in inches
SINGLE, DOUBLE = 3.5, 7.2
FLEETS = ("FD001", "FD002", "FD003", "FD004")
# one colour and one marker per family in every figure; the markers alone tell the families
# apart in greyscale
FAMILY_COLOURS = dict(zip(FAMILY_NAMES, [*SERIES, INK_SECONDARY, MUTED], strict=True))
FAMILY_MARKERS = dict(
    zip(FAMILY_NAMES, ["o", "s", "^", "v", "D", "P", "X", "*", "h", "<"], strict=True)
)
FAMILY_LABELS = {
    "ridge": "ridge",
    "knn": "k-NN",
    "forest": "forest",
    "xgboost": "XGBoost",
    "xgboost_raw": "XGBoost (raw)",
    "mlp": "MLP",
    "lstm": "LSTM",
    "tcn": "TCN",
    "cnn": "CNN",
    "transformer": "transformer",
}
# half the families get hollow markers, which separates neighbours in greyscale
HOLLOW = frozenset({"ridge", "forest", "xgboost_raw", "lstm", "cnn"})
# three settings of one quantity (lead times, tunings, selections): colour, marker and dash
VARIANTS = [
    {"color": SERIES[0], "marker": "o", "linestyle": "-"},
    {"color": SERIES[1], "marker": "s", "linestyle": "--"},
    {"color": INK_SECONDARY, "marker": "^", "linestyle": ":"},
]
TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def apply_paper_style() -> None:
    """The legacy style at paper size: 8 pt text, fonts embedded as TrueType."""
    apply_style()
    matplotlib.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def family_of(config_id: str) -> str:
    return config_id.rsplit("-", 1)[0]


def family_style(family: str, **extra) -> dict:
    return {
        "color": FAMILY_COLOURS[family],
        "marker": FAMILY_MARKERS[family],
        "label": FAMILY_LABELS[family],
    } | extra


def scatter_style(family: str) -> dict:
    """Keyword arguments of Axes.scatter for one family."""
    colour = FAMILY_COLOURS[family]
    return {
        "marker": FAMILY_MARKERS[family],
        "facecolors": "none" if family in HOLLOW else colour,
        "edgecolors": colour,
        "label": FAMILY_LABELS[family],
    }


def family_handle(family: str) -> plt.Line2D:
    """A legend entry that looks like the family's scatter marker."""
    colour = FAMILY_COLOURS[family]
    return plt.Line2D(
        [],
        [],
        linestyle="none",
        marker=FAMILY_MARKERS[family],
        markersize=4.5,
        markeredgewidth=0.7,
        markeredgecolor=colour,
        markerfacecolor="none" if family in HOLLOW else colour,
        label=FAMILY_LABELS[family],
    )


def panel_label(ax: plt.Axes, text: str) -> None:
    """A fleet or panel name above the upper left corner of the axes, clear of the data."""
    ax.text(
        0.0,
        1.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=INK,
    )


def zero_line(ax: plt.Axes, axis: str = "y") -> None:
    line = ax.axhline if axis == "y" else ax.axvline
    line(0.0, color=BASELINE, linewidth=0.8, zorder=0)


def save(fig: plt.Figure, directory: Path, name: str) -> list[Path]:
    """PDF with embedded fonts and a PNG preview."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{name}.pdf", directory / f"{name}.png"]
    fig.savefig(paths[0], metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(paths[1], dpi=200)
    plt.close(fig)
    return paths


def tex(text: object) -> str:
    """Text with LaTeX's special characters escaped."""
    return "".join(TEX_SPECIAL.get(ch, ch) for ch in str(text))


def number(value: float, decimals: int = 3) -> str:
    """A number for a table or macro; a minus sign in math mode, -- for a missing value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    text = f"{value:.{decimals}f}"
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return f"\\ensuremath{{{text}}}" if text.startswith("-") else text


def interval(lo: float, hi: float, decimals: int = 3) -> str:
    return f"[{number(lo, decimals)}, {number(hi, decimals)}]"


def booktabs(
    columns: Sequence[str],
    rows: Sequence[Sequence[str] | str],
    align: str,
    source: str,
    environment: str = "tabular",
) -> str:
    """A booktabs table body; a row given as a string is written as it is (a \\midrule, say)."""
    lines = [
        f"% {source}",
        "% written by make paper-assets; do not edit",
        f"\\begin{{{environment}}}{{{align}}}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    if environment == "longtable":
        lines += ["\\endhead", "\\bottomrule", "\\endfoot"]
    for row in rows:
        lines.append(row if isinstance(row, str) else " & ".join(row) + " \\\\")
    if environment != "longtable":
        lines.append("\\bottomrule")
    lines.append(f"\\end{{{environment}}}")
    return "\n".join(lines) + "\n"


VAL_DEFINITION = r"""% \val{<analysis>.<fleet>.<quantity>} prints a number of the paper; an undefined key prints
% ??key?? in bold, so a missing or misspelt number shows in the PDF
\makeatletter
\providecommand{\defval}[2]{\expandafter\def\csname val@#1\endcsname{#2}}
\providecommand{\val}[1]{\ifcsname val@#1\endcsname\csname val@#1\endcsname
  \else\textbf{??\texttt{\detokenize{#1}}??}\fi}
\makeatother
"""


def numbers_tex(values: Mapping[str, str], header: str) -> str:
    """paper_assets/numbers.tex: the \\val macro and one \\defval per key, sorted."""
    for key in values:
        if any(ch in key for ch in "{}\\%#$ "):
            raise ValueError(f"unsafe key {key!r}")
        if key.count(".") < 2:
            raise ValueError(f"{key!r} is not <analysis>.<fleet>.<quantity>")
    lines = [header.rstrip(), VAL_DEFINITION.rstrip(), ""]
    lines += [f"\\defval{{{key}}}{{{values[key]}}}" for key in sorted(values)]
    return "\n".join(lines) + "\n"
