#%% funcitons1.py
"""
Plotting helpers for the EV-aggregator aFRR results.

All figures use the IEEE single-column style defined in
IEEE_plotting_functions.py: figures are created with new_fig(...) and saved
with save_fig(...), so sizes, fonts, and DPI follow the IEEE template and fit
a single column (3.5 in) by default.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple

from IEEE_plotting_functions import new_fig, save_fig, IEEE

# ── Paper colour palette ──────────────────────────────────────────────────────
C_CONS     = "#2166ac"   # blue       – activation (fill + cumulative line)
C_ACT      = "#f4a582"   # pale orange – rebound (fill + cumulative line)
C_EAM      = "#1a9641"   # green      – EAM price / activation revenue
C_IMB      = "#e884e6"   # pink        – imbalance price / rebound cost / backlog
C_BID      = "#762a83"   # purple     – committed bid
C_CUM      = "#4d4d4d"   # dark grey  – cumulative profit line
C_SPOT     = "#74add1"   # light blue – spot price
C_GAIN     = "#4dac26"   # green      – spot opportunity gain

# ── Consumption-plot line colours (neutral, never blue/orange) ────────────────
C_BASE     = "#444444"   # dark grey dashed  – baseline load
C_BASE_REB = "#888888"   # medium grey solid – baseline + rebound
C_NET      = "#252525"   # near-black solid  – net load


def draw_windows(ax, windows, color=C_CUM, lw=0.6, alpha=0.4):
    """Dashed vertical lines at the start and end of each flexibility window."""
    for w in windows:
        if not w:
            continue
        start = w[0] / 4.0
        end   = (w[-1] + 1) / 4.0
        ax.axvline(start, color=color, lw=lw, alpha=alpha, linestyle="--")
        ax.axvline(end,   color=color, lw=lw, alpha=alpha, linestyle="--")


def plot_results(
    WINDOWS, results,
    EAM_scenarios_in, consumption_scenarios_in, EAM_scenarios_out,
    min_imbalance_scenarios_in, min_imbalance_scenarios_out, consumption_scenarios_out,
    w_real=None, show_bids=True, show_profit=True, show_charger=False,
    show_aggregated=True, show_aggregated_per_second=False, show_cumulative=True,
    show_revenue_breakdown=False, charger_ids=[3, 7, 12, 25],
    EAM_real=None, IMB_real=None, CONS_real=None,
):
    # === TICK SETUP (GLOBAL) ===
    major_ticks  = np.arange(0, 25, 1)
    minor_ticks  = np.arange(0, 24.25, 0.25)
    major_labels = [str(int(t)) for t in major_ticks]

    def style_time_axis(ax):
        ax.set_xticks(major_ticks)
        ax.set_xticklabels(major_labels, rotation=90, ha="center")
        ax.tick_params(axis="x", which="minor", bottom=False)

    # === UNPACK RESULTS ===
    E_real     = results["E_act_q"]
    E_reb      = results["E_reb_q"]
    a_all      = results["e_act_charger"]
    r_all      = results["e_reb_charger"]
    a_all_s    = results["p_act_sec"]
    c_bid      = results["p_bid_q"]
    lam_bid    = results["lambda_bid_q"]
    profit_act = results["profit_act_q"]
    profit_reb = results["profit_reb_q"]

    # === COMPONENT BREAKDOWNS ===
    aFRR_rev_q      = results.get("aFRR_revenue_q",   None)
    spot_opp_cost_q = results.get("spot_cost_q",      None)
    imb_cost_q      = results.get("imbalance_cost_q", None)
    spot_opp_gain_q = results.get("spot_gain_q",      None)
    spot_price      = results.get("spot_price",       None)

    # === REALIZATION ===
    if EAM_real is None or IMB_real is None or CONS_real is None:
        EAM_real  = EAM_scenarios_out[:, w_real]
        IMB_real  = min_imbalance_scenarios_out[:, w_real]
        CONS_real = consumption_scenarios_out[:, w_real, :]

    nQ  = len(E_real)
    x_q = np.arange(nQ) / 4.0

    # === IN-SAMPLE STATISTICS ===
    if EAM_scenarios_in is not None:
        EAM_scenarios_q = EAM_scenarios_in.reshape(96, 900, EAM_scenarios_in.shape[1]).mean(axis=1)
        EAM_mean = EAM_scenarios_q.mean(axis=1)
        EAM_p10  = np.percentile(EAM_scenarios_q, 10, axis=1)
        EAM_p90  = np.percentile(EAM_scenarios_q, 90, axis=1)

    if consumption_scenarios_in is not None:
        CONS_mean = consumption_scenarios_in.mean(axis=1)
        CONS_p10  = np.percentile(consumption_scenarios_in, 10, axis=1)
        CONS_p90  = np.percentile(consumption_scenarios_in, 90, axis=1)

    if min_imbalance_scenarios_in is not None:
        IMB_mean = min_imbalance_scenarios_in.mean(axis=1)
        IMB_p10  = np.percentile(min_imbalance_scenarios_in, 10, axis=1)
        IMB_p90  = np.percentile(min_imbalance_scenarios_in, 90, axis=1)
    else:
        IMB_mean = IMB_p10 = IMB_p90 = None

    # ======================================================================
    # 1. BID PLOTS
    # ======================================================================
    if show_bids:
        # -------- Plot 1: Capacity + Consumption --------
        # Height matches the combined agg-per-second + cumulative figure
        # (n_row=2 default → aspect=0.75*2=1.5 for a single-panel figure).
        fig, ax = new_fig("single", aspect=0.75)

        ax.fill_between(x_q, CONS_p10, CONS_p90, color=C_CONS, alpha=0.10,
                        label="Scenarios (p10--p90)")
        ax.plot(x_q, CONS_mean, color=C_CONS, alpha=0.35, lw=IEEE.linewidth_thin,
                label="Scenario mean")
        ax.plot(x_q, CONS_real.sum(axis=1), color=C_CONS,
                label="Realized load")
        ax.step(x_q, c_bid, where="post", color=C_BID,
                label="Capacity bid")
        ax.fill_between(x_q, 0, E_real / 0.25, step="post", color=C_BID,
                        alpha=0.25, label="Activated power")

        ax.set_xlabel("Hour of day")
        ax.set_ylabel("MW")
        ax.set_xlim(0, max(x_q))
        style_time_axis(ax)
        draw_windows(ax, WINDOWS)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

        save_fig(fig, "single", "fig_bid_capacity.pdf")
        plt.show()

        # -------- Plot 2: Prices (zoomed + full) — half the height --------
        # aspect=0.375 with n_row=2 gives total height = 3.5*0.375*2 ≈ 2.625"
        # which is half of the default two-row figure height (5.25").
        fig, (ax_zoom, ax_full) = new_fig("single", n_row=2, sharex=True, aspect=0.44)

        x_s = np.arange(EAM_real.shape[0]) / 3600.0

        def plot_prices(ax):
            ax.fill_between(x_q, EAM_p10, EAM_p90, color=C_EAM, alpha=0.12,
                            step="post", label="Scenarios (p10--p90)")
            ax.step(x_q, EAM_mean, where="post", color=C_EAM, alpha=0.35,
                    lw=IEEE.linewidth_thin, label="Scenario mean")
            ax.step(x_s, EAM_real, where="post", color=C_EAM,
                    label="Realized EAM price")
            ax.step(x_q, lam_bid, where="post", color=C_BID,
                    label="Price bid")
            draw_windows(ax, WINDOWS)
            style_time_axis(ax)
            ax.set_xlim(0, 24)

        plot_prices(ax_zoom)
        ax_zoom.set_ylim(0, 300)
        ax_zoom.set_ylabel("Price [€/MWh]")
        ax_zoom.tick_params(labelbottom=False)

        plot_prices(ax_full)
        ax_full.set_ylabel("Price [€/MWh]")
        ax_full.set_xlabel("Hour of day")

        handles, labels = ax_full.get_legend_handles_labels()
        ax_full.legend(handles, labels, loc="upper center",
                       bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False)

        save_fig(fig, "single", "fig_bid_prices_zoom_full.pdf")
        plt.show()

        # -------- Plot 3: Imbalance vs spot vs aFRR bid prices --------
        fig, ax = new_fig("single")

        ax.step(x_q, spot_price, where="post", color=C_SPOT,
                label="Spot price", zorder=3)
        if IMB_mean is not None:
            ax.fill_between(x_q, IMB_p10, IMB_p90, color=C_IMB, alpha=0.12,
                            step="post", label="Scenarios (p10--p90)")
            ax.step(x_q, IMB_mean, where="post", color=C_IMB, alpha=0.35,
                    lw=IEEE.linewidth_thin, label="Scenario mean")
        ax.step(x_q, IMB_real, where="post", color=C_IMB,
                label="Realized imbalance price", zorder=2)
        ax.step(x_q, lam_bid, where="post", color=C_BID,
                label="Price bid", zorder=3)

        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Price [€/MWh]")
        ax.set_xlim(0, 24)
        style_time_axis(ax)
        draw_windows(ax, WINDOWS)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

        save_fig(fig, "single", "fig_prices_comparison.pdf")
        plt.show()

    # ======================================================================
    # 2. PROFIT (dual y-axis, cumulative on right)
    # ======================================================================
    if show_profit:
        profit_cum = np.cumsum(profit_act - profit_reb)

        fig, ax1 = new_fig("single")
        ax2 = ax1.twinx()

        if (aFRR_rev_q is not None and spot_opp_cost_q is not None
                and imb_cost_q is not None and spot_opp_gain_q is not None):
            ax1.bar(x_q,  aFRR_rev_q,      width=0.2, color=C_EAM, alpha=0.85,
                    label="aFRR revenue")
            ax1.bar(x_q, -spot_opp_cost_q, width=0.2, color=C_IMB, alpha=0.85,
                    label="Spot opp. cost")
            ax1.bar(x_q, -imb_cost_q,      width=0.2, color=C_ACT, alpha=0.85,
                    label="Imbalance cost")
            ax1.bar(x_q,  spot_opp_gain_q, width=0.2, color=C_SPOT, alpha=0.85,
                    label="Spot opp. gain")
        else:
            ax1.bar(x_q,  profit_act, width=0.2, color=C_EAM, alpha=0.85,
                    label="Activation revenue")
            ax1.bar(x_q, -profit_reb, width=0.2, color=C_IMB, alpha=0.85,
                    label="Rebound cost")

        if show_cumulative:
            ax2.plot(x_q, profit_cum, color=C_CUM, lw=IEEE.linewidth,
                     label="Cumulative profit", zorder=5)

        ax1.axhline(0, color=C_CUM, lw=IEEE.linewidth_thin)
        ax1.set_xlabel("Hour of day")
        ax1.set_ylabel("Revenue / cost [€]")
        ax2.set_ylabel("Cumulative profit [€]")
        ax2.set_ylim(bottom=0)

        style_time_axis(ax1)
        draw_windows(ax1, WINDOWS)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center",
                   bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

        y1_min, y1_max = ax1.get_ylim()
        zero_rel = (0 - y1_min) / (y1_max - y1_min)
        cum_max  = profit_cum.max() * 1.1
        ax2.set_ylim(-zero_rel / (1 - zero_rel) * cum_max, cum_max)

        save_fig(fig, "single", "fig_profit.pdf")
        plt.show()

    # ======================================================================
    # 3. INDIVIDUAL CHARGERS (2x2 grid) -- full double-column width
    # ======================================================================
    if show_charger:
        if charger_ids is None or len(charger_ids) != 4:
            raise ValueError("charger_ids must be a list of 4 charger indices")

        fig, axes = new_fig("double", n_row=2, n_col=2, sharex=True, sharey=True)
        axes = axes.flatten()

        for ax, ch in zip(axes, charger_ids):
            base      = CONS_real[:, ch]
            act       = a_all[:, ch] / 0.25
            reb       = r_all[:, ch] / 0.25
            after_reb = base + reb
            after_all = base + reb - act

            ax.plot(x_q, base,      color=C_BASE,     lw=IEEE.linewidth_thin,
                    linestyle="--", label="Baseline")
            ax.plot(x_q, after_reb, color=C_BASE_REB, label="Baseline + rebound")
            ax.plot(x_q, after_all, color=C_NET,      label="Net load")
            ax.fill_between(x_q, base, after_reb, where=(reb > 1e-6),
                            color=C_ACT,  alpha=0.25, label="Rebound")
            ax.fill_between(x_q, after_reb, after_all, where=(act > 1e-6),
                            color=C_CONS, alpha=0.25, label="Activation")

            ax.set_title(f"Charger {ch}")
            draw_windows(ax, WINDOWS)
            style_time_axis(ax)
            for i, label in enumerate(ax.get_xticklabels()):
                if i % 2 != 0:
                    label.set_visible(False)

        for ax in axes[2:]:
            ax.set_xlabel("Hour of day")
        for ax in axes[::2]:
            ax.set_ylabel("MW")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)

        save_fig(fig, "double", "fig_charger_2x2_cons.pdf")
        plt.show()

    # ======================================================================
    # 4. AGGREGATED CONSUMPTION (quarterly)
    # ======================================================================
    if show_aggregated:
        base      = CONS_real.sum(axis=1)
        reb_MW    = r_all.sum(axis=1) / 0.25
        act_MW    = a_all.sum(axis=1) / 0.25
        after_reb = base + reb_MW
        after_all = base + reb_MW - act_MW

        fig, ax = new_fig("single")

        ax.plot(x_q, base,      color=C_BASE,     lw=IEEE.linewidth_thin,
                linestyle="--", label="Baseline")
        ax.plot(x_q, after_reb, color=C_BASE_REB, label="Baseline + rebound")
        ax.plot(x_q, after_all, color=C_NET,      label="Net load")
        ax.fill_between(x_q, base, after_reb, where=(reb_MW > 1e-6),
                        color=C_ACT,  alpha=0.25, label="Rebound")
        ax.fill_between(x_q, after_reb, after_all, where=(act_MW > 1e-6),
                        color=C_CONS, alpha=0.25, label="Activation")

        ax.set_xlabel("Hour of day")
        ax.set_ylabel("MW")
        style_time_axis(ax)
        draw_windows(ax, WINDOWS)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

        save_fig(fig, "single", "fig_agg_cons.pdf")
        plt.show()

    # ======================================================================
    # 4b + 5. AGGREGATED CONSUMPTION PER SECOND (top) +
    #         CUMULATIVE ENERGY (bottom) — shared x-axis
    # ======================================================================
    if show_aggregated_per_second:
        x_q_seconds = np.arange(96) * 900
        x_s_full    = np.arange(86400)
        x_s_hours   = x_s_full / 3600.0

        act_MW_s = a_all_s
        base     = CONS_real.sum(axis=1)
        base_s   = np.interp(x_s_full, x_q_seconds, base)
        reb_MW   = r_all.sum(axis=1) / 0.25
        reb_MW_s = np.interp(x_s_full, x_q_seconds, reb_MW)

        after_reb_s = base_s + reb_MW_s
        after_all_s = base_s + reb_MW_s - act_MW_s

        cum_act = np.cumsum(E_real)
        cum_reb = np.cumsum(E_reb)

        fig, (ax_top, ax_bot) = new_fig("single", n_row=2, sharex=True, aspect=0.44)

        # ── Top panel: per-second power ──
        ax_top.plot(x_s_hours, base_s,      color=C_BASE,     lw=IEEE.linewidth_thin,
                    linestyle="--", label="Baseline")
        #ax_top.plot(x_s_hours, after_reb_s, color=C_BASE_REB, label="Baseline + rebound")
        ax_top.plot(x_s_hours, after_all_s, lw=IEEE.linewidth_thin, color=C_NET,
                    label="Net load")
        ax_top.fill_between(x_s_hours, base_s, after_reb_s, where=(reb_MW_s > 1e-6),
                            color=C_ACT,  alpha=0.25, label="Rebound")
        ax_top.fill_between(x_s_hours, after_reb_s, after_all_s, where=(act_MW_s > 1e-6),
                            color=C_CONS, alpha=0.25, label="Activation")

        ax_top.set_ylabel("MW")
        ax_top.tick_params(labelbottom=False)
        draw_windows(ax_top, WINDOWS)

        # ── Bottom panel: cumulative energy ──
        ax_bot.plot(x_q, cum_act, color=C_CONS, label="Activation")
        ax_bot.plot(x_q, cum_reb, color=C_ACT,  label="Rebound")
        ax_bot.fill_between(x_q, cum_act, cum_reb, where=cum_reb < cum_act,
                            color=C_IMB, alpha=0.25, label="Unrecovered backlog")

        ax_bot.set_xlabel("Hour of day")
        ax_bot.set_ylabel("MWh")
        ax_bot.set_xlim(0, 24)
        style_time_axis(ax_bot)
        draw_windows(ax_bot, WINDOWS)

        # Shared legend: collect unique handles from both panels
        handles_top, labels_top = ax_top.get_legend_handles_labels()
        handles_bot, labels_bot = ax_bot.get_legend_handles_labels()
        seen = set()
        combined_h, combined_l = [], []
        # Bottom-panel handles first so cumulative lines represent Activation/Rebound
        for h, l in zip(handles_bot + handles_top, labels_bot + labels_top):
            if l not in seen:
                seen.add(l)
                combined_h.append(h)
                combined_l.append(l)

        # For Activation and Rebound: combine the solid line with its fill patch
        # so the legend icon shows both (transparent swatch + solid line).
        handler_map = {}
        for i, l in enumerate(combined_l):
            if l == "Activation":
                combined_h[i] = (
                    mpatches.Patch(facecolor=C_CONS, alpha=0.25, edgecolor="none"),
                    mlines.Line2D([], [], color=C_CONS, lw=IEEE.linewidth),
                )
                handler_map[combined_h[i]] = HandlerTuple(ndivide=None, pad=0.5)
            elif l == "Rebound":
                combined_h[i] = (
                    mpatches.Patch(facecolor=C_ACT, alpha=0.25, edgecolor="none"),
                    mlines.Line2D([], [], color=C_ACT, lw=IEEE.linewidth),
                )
                handler_map[combined_h[i]] = HandlerTuple(ndivide=None, pad=0.5)

        ax_bot.legend(combined_h, combined_l, loc="upper center",
                      bbox_to_anchor=(0.5, -0.33), ncol=3, frameon=False,
                      handler_map=handler_map)

        save_fig(fig, "single", "fig_agg_cons_per_second.pdf")
        plt.show()


def check_violations(results, consumption_scenarios_out, w_real, P_MAX, N_charge):
    """Check per-charger consumption/power/overlap violations after realization."""
    violations = []
    tol = 1e-6

    CONS_real  = consumption_scenarios_out[:, w_real, :]
    a_q_c_all  = results["e_act_charger"]
    r_q_c_all  = results["e_reb_charger"]

    for n in range(len(N_charge)):
        for q in range(96):
            cons        = CONS_real[q, n]
            act         = a_q_c_all[q, n] / 0.25
            reb         = r_q_c_all[q, n] / 0.25
            total_after = cons - act + reb

            if total_after < -tol:
                violations.append((n, q, "Negative total consumption", total_after))
            if total_after > P_MAX + tol:
                violations.append((n, q, "Above max power", total_after))
            if act > tol and reb > tol:
                violations.append((n, q, "Activation and rebound overlap", (act, reb)))

    return violations
# %%
