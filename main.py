"""Generate the convex and nonconvex panels of Fig. 1."""

import argparse
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BASE_FS = 34
AXIS_FS = 42
TICK_FS = 34
LEGEND_FS = 26
ANNO_FS = 26
STAR_MS = 22
DOT_MS = 18
KKT_MS = 20


def configure_style(usetex=False):
    params = {
        "text.usetex": usetex,
        "font.family": "serif",
        "font.size": BASE_FS,
        "axes.labelsize": AXIS_FS,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "legend.fontsize": LEGEND_FS,
        "lines.linewidth": 2.8,
        "axes.linewidth": 1.8,
    }
    if usetex:
        params["text.latex.preamble"] = (
            r"\usepackage{amsmath}\usepackage{amssymb}"
            r"\usepackage{newtxtext}\usepackage{newtxmath}"
        )
        if any(shutil.which(name) for name in ("gs", "gswin64c", "gswin32c")):
            params.update({"ps.usedistiller": "ghostscript", "ps.distiller.res": 6000})
    else:
        params.update({"font.serif": ["STIXGeneral"], "mathtext.fontset": "stix"})
    plt.rcParams.update(params)


def configure_axes(ax, legend_location):
    ax.set_xlim(-2.8, 1.8)
    ax.set_ylim(-2.8, 1.8)
    ax.set_xticks([-2, -1, 0, 1])
    ax.set_yticks([-2, -1, 0, 1])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.legend(
        loc=legend_location, frameon=True, framealpha=1.0,
        borderpad=0.35, handlelength=1.4,
    )
    ax.grid(False)
    ax.figure.subplots_adjust(left=0.10, right=0.995, bottom=0.10, top=0.995)


def add_equilibrium_markers(ax, multiplier):
    ax.plot(
        -1.6, -1.6, "b*", markersize=STAR_MS,
        label=r"$x_{\min}=(-1.6,-1.6)$",
    )
    ax.plot(
        -1.0, -1.0, "ro", markersize=DOT_MS,
        label=r"$x^\circ=(-1,-1)$",
    )
    ax.annotate(
        r"$\mathbf{Global\ vGNE}$" + "\n"
        r"$x^\circ=(-1,-1)$" + "\n"
        + rf"$\lambda^\circ={multiplier:g}$",
        xy=(-1.0, -1.0), xytext=(-0.35, -2.40),
        arrowprops=dict(
            facecolor="black", edgecolor="black", shrink=0.06,
            width=1.8, headwidth=9,
        ),
        fontsize=ANNO_FS,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black"),
    )


def local_nonconvex_constraint(x):
    return 0.5 * (x**2 - 5.0 / 8.0)**2 - 9.0 / 128.0


def convex_panel(x1, X1, X2, potential, contour_levels):
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    feasible = X1 + X2 + 2.0 >= 0
    ax.contourf(X1, X2, feasible, levels=[0.5, 1.5], colors=["0.90"])
    ax.contour(X1, X2, potential, levels=contour_levels, cmap="viridis")
    ax.plot(x1, -x1 - 2.0, "k--", label=r"$-x_1-x_2-2\leqslant0$")
    add_equilibrium_markers(ax, 0.75)
    configure_axes(ax, "upper right")
    return fig


def nonconvex_panel(X1, X2, potential, contour_levels):
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    constraint = local_nonconvex_constraint(X1) + local_nonconvex_constraint(X2)
    ax.contourf(
        X1, X2, constraint <= 0, levels=[0.5, 1.5], colors=["0.90"],
    )
    ax.contour(X1, X2, potential, levels=contour_levels, cmap="viridis")
    ax.contour(X1, X2, constraint, levels=[0], colors="k", linestyles="--")
    ax.plot([], [], "k--", label=r"$\sum_{i=1}^2 g_i(x_i)\leqslant0$")
    add_equilibrium_markers(ax, 1)
    ax.plot(
        0.5, 0.5, "rX", markersize=KKT_MS,
        label=r"$x_{\mathrm{KKT}}=(0.5,0.5)$",
    )
    ax.annotate(
        r"$\mathbf{Nonglobal\ KKT}$" + "\n"
        r"$x_{\mathrm{KKT}}=(0.5,0.5)$" + "\n"
        r"$\lambda_{\mathrm{KKT}}=7$",
        xy=(0.5, 0.5), xytext=(1.68, 1.68),
        ha="right", va="top", multialignment="left",
        arrowprops=dict(
            facecolor="red", edgecolor="red", shrink=0.06,
            width=1.8, headwidth=9,
        ),
        fontsize=ANNO_FS,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="red"),
    )
    configure_axes(ax, "upper left")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "results" / "figure1",
        help="Output folder (default: results/figure1 beside this script).",
    )
    parser.add_argument(
        "--usetex", action="store_true",
        help="Use LaTeX and the manuscript fonts instead of built-in math rendering.",
    )
    args = parser.parse_args()
    if args.usetex and not all(shutil.which(name) for name in ("latex", "dvips", "dvipng")):
        parser.error("--usetex requires latex, dvips, and dvipng on PATH.")
    configure_style(args.usetex)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    x1 = np.linspace(-3.0, 2.0, 1000)
    x2 = np.linspace(-3.0, 2.0, 1000)
    X1, X2 = np.meshgrid(x1, x2)
    potential = 0.5 * (X1**2 + X2**2) + 2.0 * (X1 + X2) + 0.25 * X1 * X2
    contour_levels = np.linspace(-3.1, 9.5, 25)

    panels = {
        "convex_case": convex_panel(x1, X1, X2, potential, contour_levels),
        "nonconvex_case": nonconvex_panel(X1, X2, potential, contour_levels),
    }
    for stem, fig in panels.items():
        for extension in ("eps", "png"):
            path = output_dir / f"{stem}.{extension}"
            fig.savefig(path, format=extension, dpi=600, bbox_inches="tight", pad_inches=0.02)
            print(f"Saved: {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
