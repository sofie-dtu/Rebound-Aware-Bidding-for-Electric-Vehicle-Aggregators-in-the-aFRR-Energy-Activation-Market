#%%
import matplotlib as mpl
from matplotlib import font_manager

font_path = None
font_manager.fontManager.addfont(font_path)
lmroman = font_manager.FontProperties(fname=font_path)

mpl.rcParams.update({
    "font.family": lmroman.get_name(),
    "mathtext.fontset": "cm",
    "font.size": 22,
    "axes.labelsize": 22,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})
from datetime import datetime, timedelta

def quarter_to_time(q):
    start = datetime(2000, 1, 1, 0, 0)
    return (start + timedelta(minutes=15*q)).strftime("%H:%M")
import matplotlib.pyplot as plt

def plot_three_panel_figure(WINDOWS, win_id, q0):

    fig, axes = plt.subplots(
        2, 1,
        figsize=(16, 6),
        sharex=False,
        gridspec_kw={"height_ratios": [1,1]},
    )

    # ====================================================
    # 1) FULL DAY WINDOWS
    # ====================================================
    # ====================================================
    # 1) FULL DAY WINDOWS — SINGLE LINE
    # ====================================================
    ax = axes[0]
    colors = plt.get_cmap("tab10")

    y = 0  # single timeline

    for i, win in enumerate(WINDOWS):
        ax.plot(
            [win[0], win[-1] + 1],
            [y, y],
            linewidth=12,
            solid_capstyle="butt",
            label=f"Window {i+1}",
            color = colors(i)
        )

    ax.set_yticks([])
    ax.set_ylim(-1, 1)

    ticks = list(range(0, 96, 9))
    ax.set_xticks(ticks)
    ax.set_xticklabels([quarter_to_time(q) for q in ticks], rotation=45)

    ax.set_title("Windows Across Full Day")
    ax.set_xlabel("Time of day")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.7)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.7),
        ncol=len(WINDOWS),
        frameon=False
    )


    # ====================================================
    # 2–3) PROBLEM TIMELINES
    # ====================================================
    def problem_axis(ax, q0_local):

        Q_win = WINDOWS[win_id]

        q_opt = q0_local - 2
        Q_realized = [q for q in Q_win if q < q0_local - 2]
        Q_prev2    = [q for q in [q0_local - 2, q0_local - 1] if q in Q_win]
        Q_solve    = [q for q in Q_win if q >= q0_local]

        split = int(0.8 * len(Q_win))
        Q_enf, Q_tail = Q_win[:split], Q_win[split:]

        def block(Q, label):
            if Q:
                ax.plot(
                    [Q[0], Q[-1] + 1], [0, 0],
                    linewidth=12,
                    solid_capstyle="butt",
                    label=label,
                    color={"Q_realized": "tab:purple", "Q_prev2": "tab:brown", "Q_solve": "tab:pink"}[label])



        # Background
        if Q_enf:
            ax.axvspan(Q_enf[0], Q_enf[-1]+1, alpha=0.1, color="tab:blue", label="Q_enf")
        if Q_tail:
            ax.axvspan(Q_tail[0], Q_tail[-1]+1, alpha=0.1, color="tab:red", label="Q_tail")

        block(Q_realized, "Q_realized")
        block(Q_prev2, "Q_prev2")
        block(Q_solve, "Q_solve")

        ax.axvline(q_opt, linestyle=":", color="black", linewidth=1.5,
                   label=r"Optimization time $q_0-2\:(30\:\text{min})$")
        # add line 25 min before q0
        ax.axvline(q0_local - 1.67, linestyle="--", color="gray", linewidth=1.5,
                   label=r"Bid submission time ($25\:\text{min}$ before $q_0$)")
        ax.axvline(q0_local, linestyle="--", color="black", linewidth=1.5,
                   label=r"Target time $q_0$")

        ax.set_xticks(Q_win)
        ax.set_xticklabels([quarter_to_time(q) for q in Q_win], rotation=60)
        ax.set_yticks([])
        ax.set_ylim(-1, 1)
        ax.grid(True, axis="x", linestyle=":", linewidth=0.6)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.7),
            ncol=4,
            frameon=False
        )

    problem_axis(axes[1], q0)
    axes[1].set_title(f"Window {win_id+1} — target $q_0={q0}$")

    axes[1].set_xlabel("Time of day")


    fig.subplots_adjust(hspace=0.9)



    plt.show()
    # save as pdf
    fig.savefig("three_panel_figure.pdf", bbox_inches="tight")
WINDOWS = [
    list(range(0, 24)),
    list(range(24, 32)),
    list(range(32, 61)),
    list(range(61, 96))
]

plot_three_panel_figure(WINDOWS, win_id=0, q0=10)

# %%
