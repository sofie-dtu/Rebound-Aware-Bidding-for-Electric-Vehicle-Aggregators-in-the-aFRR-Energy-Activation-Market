# %%
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scienceplots

# ── Column widths (inches) 
SINGLE_COL = 3.5    # one-column figure
DOUBLE_COL = 7.16   # two-column figure / full-width figure

# ── Aspect ratios 
# Classic "golden ratio"-ish height for a single panel
ASPECT = 0.75


class IEEE:
    """Central place for every IEEE figure dimension and style constant."""

    # --- Column widths ---
    col1 = SINGLE_COL
    col2 = DOUBLE_COL

    # --- Font sizes (pt) ---
    fontsize       = 10     # body / tick labels
    fontsize_small = 8      # secondary labels, legend
    fontsize_title = 8      # subplot / figure title (same as body per IEEE)

    # --- Line & marker weights ---
    linewidth      = 1.0    # data lines
    linewidth_thin = 0.6    # grid, axes spines
    markersize     = 3.5    # scatter / line markers
    capsize        = 2.5    # error-bar cap length

    # --- Layout padding ---
    pad_inches     = 0.01   # tight-layout border around the whole figure

    # --- DPI ---
    dpi_screen     = 150    # preview
    dpi_print      = 600    # final PDF/EPS for submission


# ── Apply global rcParams ─────────────────────────────────────────────────────
def _latex_available():
    """True if a `latex` executable is on PATH (needed for text.usetex)."""
    import shutil
    return shutil.which("latex") is not None


def apply_ieee_style(usetex=None):
    """
    Apply the IEEE single-column style.

    usetex : bool or None
        Whether to render text with a real LaTeX install. If None (default),
        it is enabled only when a `latex` binary is found on PATH; otherwise
        matplotlib's built-in mathtext is used, so no LaTeX install is needed.
    """
    if usetex is None:
        usetex = _latex_available()

    plt.style.use(['science', 'ieee'])
    sns.set_theme(context="paper", style="ticks")
    sns.set(rc={'axes.facecolor':'#f4f4f4', 'figure.facecolor':'white', 'grid.color':'#fff'})
    sns.set_context("notebook", rc={"grid.linewidth": 0.5})

    mpl.rcParams.update({
        # --- Font ---
        "text.usetex"        : usetex,
        "mathtext.fontset"   : "cm",   # used when usetex is False
        "font.family"        : "serif",
        "font.serif"         : ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size"          : IEEE.fontsize,
        "axes.titlesize"     : IEEE.fontsize_title,
        "axes.labelsize"     : IEEE.fontsize,
        "xtick.labelsize"    : IEEE.fontsize_small,
        "ytick.labelsize"    : IEEE.fontsize_small,
        "legend.fontsize"    : IEEE.fontsize_small,
        "legend.title_fontsize": IEEE.fontsize_small,
        "figure.titlesize"   : IEEE.fontsize,

        # --- Lines & markers ---
        "lines.linewidth"    : IEEE.linewidth,
        "lines.markersize"   : IEEE.markersize,
        "patch.linewidth"    : IEEE.linewidth_thin,

        # --- Axes ---
        "axes.linewidth"     : IEEE.linewidth_thin,
        "axes.grid"          : True,
        "grid.linewidth"     : 0.4,
        "grid.alpha"         : 0.4,
        "axes.axisbelow"     : True,   # grid behind data
        "axes.edgecolor"     : "grey",

        # --- Ticks ---
        "xtick.major.width"  : IEEE.linewidth_thin,
        "ytick.major.width"  : IEEE.linewidth_thin,
        "xtick.minor.width"  : 0.4,
        "ytick.minor.width"  : 0.4,
        "xtick.major.size"   : 3.0,
        "ytick.major.size"   : 3.0,
        "xtick.minor.size"   : 1.5,
        "ytick.minor.size"   : 1.5,
        "xtick.direction"    : "in",
        "ytick.direction"    : "in",
        "xtick.bottom"       : True,
        "ytick.left"         : True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.color"        : "grey",
        "xtick.labelcolor"   : "black",
        "ytick.color"        : "grey",
        "ytick.labelcolor"   : "black",

        # --- Legend ---
        "legend.framealpha"  : 0.9,
        "legend.edgecolor"   : "0.8",
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,

        # --- Figure ---
        "figure.dpi"         : IEEE.dpi_screen,
        "savefig.dpi"        : IEEE.dpi_print,
        "savefig.bbox"       : "tight",
        "savefig.pad_inches" : IEEE.pad_inches,

        # --- PDF/PS backend (vector output) ---
        "pdf.fonttype"       : 42,   # embed TrueType (avoids Type 3 fonts)
        "ps.fonttype"        : 42,
    })

# Apply immediately on import
apply_ieee_style()

# %%

# ── Core helper: resize figure so every axes box hits its target size 
def set_size_from_axes(
    fig: plt.Figure,
    columns: str,
    ax_w: float,
    ax_h: float,
    axes=None,
    r: float = 1.0, 
) -> None:
    
    if axes is None:
        axes = fig.axes[0]
 
    fig.tight_layout(pad=0.3)
 
    pos = axes.get_position()
 
    base_w = SINGLE_COL if columns == "single" else DOUBLE_COL
    fig_w_new = base_w * r   # ← apply r here instead of using bare column width
    fig_h_new = ax_h / pos.height
 
    fig.set_size_inches(fig_w_new, fig_h_new)
    fig.tight_layout(pad=0.3)
 
 
# ── Figure factory 
def new_fig(
    columns: str = "single",
    n_row: int = 1,
    n_col: int = 1,
    aspect: float | None = None,
    ax_aspect: float | None = None,
    sharex: bool = False,
    sharey: bool = False,
    r: float = 1.0,
):
    if columns not in ("single", "double"):
        raise ValueError("columns must be 'single' or 'double'")
 
    fig_w = SINGLE_COL * r if columns == "single" else DOUBLE_COL * r
    ax_box_w = fig_w / n_col
 
    if ax_aspect is not None:
        ax_box_h = ax_box_w * ax_aspect
        fig_h_init = ax_box_h * n_row
    else:
        fig_h_init = fig_w * (aspect if aspect is not None else ASPECT)
        if n_row > 1:
            fig_h_init *= n_row
 
    subplot_kw = dict(box_aspect=1) if ax_aspect is not None else {}
    fig, axes_out = plt.subplots(
        nrows=n_row, ncols=n_col,
        figsize=(fig_w, fig_h_init),
        sharex=sharex,
        sharey=sharey,
        subplot_kw=subplot_kw if subplot_kw else None,
        layout="constrained",
    )
 
    # Store all three so save_fig() can reconstruct the correct size
    fig._ieee_ax_aspect = ax_aspect
    fig._ieee_ax_box_w  = ax_box_w
    fig._ieee_r         = r

    return fig, axes_out
 
 
# ── Save helper 
def save_fig(fig: plt.Figure, columns: str, path: str, **kwargs) -> None:
    ax_aspect = getattr(fig, "_ieee_ax_aspect", None)
    ax_box_w  = getattr(fig, "_ieee_ax_box_w",  None)
    r         = getattr(fig, "_ieee_r",          1.0)  # ← read r (default 1 for back-compat)
 
    if ax_aspect is not None and ax_box_w is not None:
        set_size_from_axes(
            fig, columns,
            ax_w=ax_box_w,
            ax_h=ax_box_w * ax_aspect,
            r=r,                       # ← forward r
        )
    else:
        fig.tight_layout(pad=0.3)
 
    fig.savefig(path, bbox_inches="tight", pad_inches=IEEE.pad_inches, **kwargs)
    print(f"Saved → {path}")
 
 
# Demo / sanity-check  (python ieee_plot_config.py)
if __name__ == "__main__":
    ev_color = '#cad2c5'
    wind_color = '#52796f'
    conv_color = '#2f3e46'

    def get_eps_price(eps, g):
        generator_cost_intercept = {'g1': 4.5, 'g2': 3.0, 'g3': 10.0}
        generator_cost_slope = {'g1': 2.5, 'g2': 2.0, 'g3': 0}
        return generator_cost_intercept[f"g{g}"] - generator_cost_slope[f"g{g}"]*eps/0.2

    x = np.linspace(0, 0.2, 100)
  
    fig, ax = new_fig("single", n_row=1, n_col=1, sharey=True, sharex=False, ax_aspect=1, aspect=0.8, r=0.5)
    # ax.set_box_aspect(1)
    ax.plot(1-x, get_eps_price(x, 3), color=conv_color, linestyle="solid", lw=3, alpha=0.9, label='Conventional')
    ax.plot(1-x, get_eps_price(x, 1), color=wind_color, lw=3, alpha=0.9, label='EV')
    ax.plot(1-x, get_eps_price(x, 2), color=ev_color, linestyle="solid", lw=3, alpha=0.9, label='Wind')
    ax.set_xlabel(fr'$1-\varepsilon_t$')
    ax.set_ylabel(u'Cost [\u20ac/kW]')
    ax.legend()
    # save_fig(fig, "single", "Latex\\686fa94c623cde79680deee6\\05_1col_2subplots_v.pdf")
    # plt.close()