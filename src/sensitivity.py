#%% sensitivity.py
"""
Sensitivity analysis for the EV-aggregator aFRR bidding model.

Two entry points:
    run_sensitivity         -- sweep a single model parameter (alpha, theta).
    sensitivity_scenarios   -- sweep the number of in-sample scenarios n_omega.

Both run every out-of-sample realization for each setting, aggregate the
results (mean +/- std), and draw single-column IEEE summary figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bidding_model import run_fleet_model
from funcitons1 import plot_results
from IEEE_plotting_functions import new_fig, save_fig


def plot_sensitivity_summary(df, x, x_label):
    """Draw mean +/- std summary curves vs. the swept parameter `x`."""

    def _band_plot(y_mean, y_std, color, ylabel, fname, marker="o-", label=None):
        fig, ax = new_fig("single")
        ax.fill_between(df[x], df[y_mean] - df[y_std], df[y_mean] + df[y_std],
                        alpha=0.2, color=color)
        ax.plot(df[x], df[y_mean], marker, color=color, label=label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        if label is not None:
            ax.legend()
        save_fig(fig, "single", fname)
        plt.show()

    # Combined figure: Profit (top) + Activation/Rebound (bottom)
    fig, (ax_profit, ax_energy) = new_fig("single", n_row=2, sharex=True, aspect=0.44)
    
    # Top plot: Profit
    ax_profit.fill_between(df[x], df["profit_mean"] - df["profit_std"],
                           df["profit_mean"] + df["profit_std"],
                           alpha=0.2, color="tab:green")
    ax_profit.plot(df[x], df["profit_mean"], "o-", color="tab:green", label="Profit")
    ax_profit.set_ylabel("Profit [€]")
    ax_profit.legend(loc="best")
    
    # Bottom plot: Activation + rebound (both on same axes)
    ax_energy.fill_between(df[x], df["activation_mean"] - df["activation_std"],
                           df["activation_mean"] + df["activation_std"],
                           alpha=0.2, color="tab:blue")
    ax_energy.plot(df[x], df["activation_mean"], "o-", color="tab:blue", label="Activation")
    ax_energy.fill_between(df[x], df["rebound_mean"] - df["rebound_std"],
                           df["rebound_mean"] + df["rebound_std"],
                           alpha=0.2, color="tab:orange")
    ax_energy.plot(df[x], df["rebound_mean"], "s--", color="tab:orange", label="Rebound")
    ax_energy.set_xlabel(x_label)
    ax_energy.set_ylabel("Energy [MWh]")
    ax_energy.legend(loc="best")
    
    save_fig(fig, "single", "fig_sensitivity_profit_energy.pdf")
    plt.show()

    # 3) Violations
    _band_plot("viol_frac_mean", "viol_frac_std", "tab:red",
               "Violation Fraction [%]", "fig_sensitivity_violations.pdf")

    # 4) Runtime
    if "runtime_mean" in df.columns:
        _band_plot("runtime_mean", "runtime_std", "tab:purple",
                   "Runtime [min]", "fig_sensitivity_runtime.pdf")
    else:
        print("Warning: runtime_mean not found in df -- runtime not plotted.")

    # 5) Average backlog across windows
    if "backlog_avg_mean" in df.columns:
        _band_plot("backlog_avg_mean", "backlog_avg_std", "tab:blue",
                   "Average backlog [MWh]", "fig_sensitivity_backlog_avg.pdf")


def plot_sensitivity_multiday(param_values, per_day_records, day_labels, param_name):
    """
    Draw one line per day for a multi-day sensitivity sweep.

    per_day_records[i][j] = KPI dict for param_values[i], day j.
    A thick black mean line is added on top of the per-day lines.
    """
    label_map = {
        "alpha": r"$\alpha$",
        "theta": r"$\theta$",
        "W": "In-sample scenarios $|\\Omega|$",
    }
    x_label = label_map.get(param_name, param_name)
    n_days = len(day_labels)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_days))

    def _extract(metric):
        return np.array([[per_day_records[i][j][metric] for j in range(n_days)]
                         for i in range(len(param_values))])

    profit     = _extract("profit")
    activation = _extract("activation")
    rebound    = _extract("rebound")
    viol       = _extract("viol_frac")

    # --- Combined: Profit (top) + Activation/Rebound (bottom) ---
    fig, (ax_p, ax_e) = new_fig("single", n_row=2, sharex=True, aspect=0.44)

    day_handles = []
    for j, (label, color) in enumerate(zip(day_labels, colors)):
        h, = ax_p.plot(param_values, profit[:, j],     "o-", color=color, linewidth=1)
        ax_e.plot(param_values, activation[:, j], "o-", color=color, linewidth=1)
        ax_e.plot(param_values, rebound[:, j],    "s--", color=color, linewidth=1)
        day_handles.append((h, label))

    ax_p.set_ylabel("Profit [€]")
    ax_e.set_xlabel(x_label)
    ax_e.set_ylabel("Energy [MWh]")

    handles, labels = zip(*day_handles)
    ax_e.legend(handles, labels, loc="upper center",
                bbox_to_anchor=(0.5, -0.30), ncol=n_days, frameon=False)

    save_fig(fig, "single", "fig_sensitivity_multiday_profit_energy.pdf")
    plt.show()

    # --- Violations (same color scheme, legend shown once below) ---
    fig_v, ax_v = new_fig("single")
    viol_handles = []
    for j, (label, color) in enumerate(zip(day_labels, colors)):
        h, = ax_v.plot(param_values, viol[:, j], "o-", color=color, linewidth=1)
        viol_handles.append((h, label))
    ax_v.set_xlabel(x_label)
    ax_v.set_ylabel("Violation Fraction [%]")
    handles_v, labels_v = zip(*viol_handles)
    ax_v.legend(handles_v, labels_v, loc="upper center",
                bbox_to_anchor=(0.5, -0.18), ncol=n_days, frameon=False)
    save_fig(fig_v, "single", "fig_sensitivity_multiday_violations.pdf")
    plt.show()


def _collect_run_stats(res, CONS_test, test_idx):
    """
    Per-realization KPIs extracted from one model result.
    
    CONS_test can be either:
      - 3D array (scenarios): shape (T, W, N_charge) -- uses test_idx to index
      - 2D array (actual):    shape (T, N_charge)    -- test_idx is ignored
    """
    if CONS_test.ndim == 3:
        # Out-of-sample scenarios: index by test_idx
        CONS_real_q = CONS_test[:, test_idx, :].sum(axis=1)
    else:
        # Actual realized data: use directly
        CONS_real_q = CONS_test.sum(axis=1)
    
    viol_frac = np.mean(res["p_bid_q"] > CONS_real_q) * 100.0
    return {
        "activation": res["E_act_q"].sum(),
        "rebound": res["E_reb_q"].sum(),
        "profit": res["profit_net_total"],
        "viol_frac": viol_frac,
        "runtime": res["runtime_min"],
        "backlog_avg": res["backlog_end_window"].mean(),
    }


def _aggregate(run_data, extra=None):
    """Mean/std aggregation of per-realization KPIs into one summary row."""
    df = pd.DataFrame(run_data)
    row = dict(extra or {})
    for key in ("activation", "rebound", "profit", "viol_frac", "runtime", "backlog_avg"):
        row[f"{key}_mean"] = df[key].mean()
        row[f"{key}_std"] = df[key].std()
    return row


def sensitivity_scenarios(
    W_values,
    sc_master,
    base_args_template,
    plot_individual=False,
    show_summary=True,
    use_actual=False,
    scale_in=1.0,
    scale_out=1.0,
    W=None,
):
    """
    Sensitivity over the number of in-sample scenarios n_omega.
    
    If use_actual=False: The same out-of-sample scenarios are used for every setting.
    If use_actual=True: The actual realized data is used for evaluation.
    
    W: fixed number of in-sample scenarios to use (only for single runs, not sweeps).
       When sweeping W_values, leave W=None to vary it dynamically.
    scale_in, scale_out: normalization factors to apply to CONS arrays.
    """
    summaries = []
    all_results = []
    
    if use_actual:
        # Use actual realized data
        EAM_test = sc_master["EAM_actual"]
        IMB_test = sc_master["IMB_actual"]
        CONS_test = sc_master["CONS_actual"] * scale_out  # Apply scaling
        N_test_realizations = 1
    else:
        # Use out-of-sample scenarios
        EAM_test = sc_master["EAM_out"]
        IMB_test = sc_master["IMB_out"]
        CONS_test = sc_master["CONS_out"] * scale_out  # Apply scaling
        N_test_realizations = CONS_test.shape[1]
    
    N_charge = range(CONS_test.shape[-1]) if use_actual else range(CONS_test.shape[2])

    for i, W_val in enumerate(W_values):
        print(f"\n=== Scenario sensitivity {i+1}/{len(W_values)}: n_omega = {W_val} ===")

        # Slice in-sample scenarios for this W value
        EAM_in_w = sc_master["EAM_in"][:, :W_val]
        IMB_in_w = sc_master["IMB_in"][:, :W_val]
        CONS_in_w = sc_master["CONS_in"][:, :W_val, :].sum(axis=2) * scale_in
        
        # Get actual number of scenarios available (in case W_val exceeds available)
        n_omega_actual = EAM_in_w.shape[1]
        
        if n_omega_actual < W_val:
            print(f"  Warning: Only {n_omega_actual} scenarios available, requested {W_val}")

        base_args = base_args_template.copy()
        base_args.update(dict(
            EAM_scenarios=EAM_in_w,
            min_imbalance_scenarios=IMB_in_w,
            consumption_scenarios=CONS_in_w,
            N_charge=N_charge,
            n_omega=n_omega_actual,
        ))

        if not use_actual:
            base_args.update(dict(
                EAM_scenarios_out=EAM_test,
                min_imbalance_scenarios_out=IMB_test,
                consumption_scenarios_out=CONS_test,
            ))

        run_data = []
        for test_idx in range(N_test_realizations):
            if use_actual:
                print(f"  Running on actual realized data")
                args = base_args.copy()
                args.update(dict(
                    EAM_real=EAM_test,
                    IMB_real=IMB_test,
                    consumption_real=CONS_test,
                ))
            else:
                print(f"  OOS realization {test_idx+1}/{N_test_realizations}")
                args = base_args.copy()
                args["w_real"] = test_idx
            
            res = run_fleet_model(**args)
            all_results.append(res)
            run_data.append(_collect_run_stats(res, CONS_test, test_idx if not use_actual else 0))

            if plot_individual:
                plot_results(
                    base_args["WINDOWS"], res,
                    EAM_in_w, CONS_in_w, EAM_test, IMB_in_w, IMB_test, CONS_test,
                    w_real=test_idx if not use_actual else None,
                    show_bids=True, show_profit=True,
                    show_aggregated=True, show_cumulative=True,
                    show_revenue_breakdown=True,
                )

        summaries.append(_aggregate(run_data, extra={"W": W_val}))

    df_summary = pd.DataFrame(summaries)
    if show_summary:
        plot_sensitivity_summary(df_summary, x="W", x_label="In-sample scenarios $|\\Omega|$")

    return df_summary, all_results


def run_sensitivity_multiday(
    param_name,
    param_values,
    day_data,
    base_args_template,
    plot_individual=False,
    show_summary=True,
    W=None,
):
    """
    Sensitivity sweep over a single parameter averaged across multiple days.

    For each (param_value, day) pair the model is re-solved using that day's own
    in-sample scenarios (planning step) and that day's own realized prices and
    consumption (realization step).  KPIs are then averaged across days so that
    day-to-day variability does not confound the parameter effect.

    day_data : list of dicts, one per held-out day, each with keys:
        sc_master  -- pipeline output for that day
        scale_in   -- normalization factor for in-sample CONS
        scale_out  -- normalization factor for actual CONS
        date       -- (optional) string label used in progress messages
    """
    summaries = []
    all_results = []
    per_day_records = []   # per_day_records[i][j] = KPI dict for param_values[i], day j

    for i, value in enumerate(param_values):
        print(f"\n=== Sensitivity {param_name} = {value}  ({i+1}/{len(param_values)}) ===")
        run_data = []

        for day_idx, dd in enumerate(day_data):
            sc = dd["sc_master"]
            s_in = dd["scale_in"]
            s_out = dd["scale_out"]
            spot_price = sc["SPOT_actual"]

            EAM_in = sc["EAM_in"] if W is None else sc["EAM_in"][:, :W]
            IMB_in = sc["IMB_in"] if W is None else sc["IMB_in"][:, :W]
            CONS_in = (
                sc["CONS_in"] if W is None else sc["CONS_in"][:, :W, :]
            ).sum(axis=2) * s_in

            EAM_real = sc["EAM_actual"]
            IMB_real = sc["IMB_actual"]
            CONS_real = sc["CONS_actual"] * s_out

            N_charge = range(CONS_real.shape[-1])
            n_omega = EAM_in.shape[1]

            args = base_args_template.copy()
            args.update({
                param_name: value,
                "spot_price": spot_price,
                "EAM_scenarios": EAM_in,
                "min_imbalance_scenarios": IMB_in,
                "consumption_scenarios": CONS_in,
                "N_charge": N_charge,
                "n_omega": n_omega,
                "EAM_real": EAM_real,
                "IMB_real": IMB_real,
                "consumption_real": CONS_real,
            })

            print(f"  Day {day_idx+1}/{len(day_data)}: {dd.get('date', '')}")
            res = run_fleet_model(**args)
            all_results.append(res)
            run_data.append(_collect_run_stats(res, CONS_real, 0))

            if plot_individual:
                plot_results(
                    args["WINDOWS"], res,
                    EAM_in, CONS_in, EAM_real, IMB_in, IMB_real, CONS_real,
                    show_bids=True, show_profit=True,
                    show_aggregated=True, show_cumulative=True,
                )

        per_day_records.append(run_data)
        summaries.append(_aggregate(run_data, extra={param_name: value}))

    df_summary = pd.DataFrame(summaries)

    if show_summary:
        day_labels = [dd.get("date", f"Day {j+1}") for j, dd in enumerate(day_data)]
        plot_sensitivity_multiday(param_values, per_day_records, day_labels, param_name)

    return df_summary, all_results


def run_sensitivity(
    param_name,
    param_values,
    sc_master,
    base_args_template,
    plot_individual=False,
    show_summary=True,
    use_actual=False,
    scale_in=1.0,
    scale_out=1.0,
    W=None,
):
    """
    Generic sensitivity sweep over a single model parameter (e.g. alpha, theta).
    
    If use_actual=False: Tests against out-of-sample scenarios.
    If use_actual=True: Tests against actual realized data.
    
    W: number of in-sample scenarios to use (if None, uses all available)
    scale_in, scale_out: normalization factors to apply to CONS arrays.
    """
    summaries = []
    all_results = []

    # Extract and optionally slice the in-sample scenarios
    EAM_in = sc_master["EAM_in"] if W is None else sc_master["EAM_in"][:, :W]
    IMB_in = sc_master["IMB_in"] if W is None else sc_master["IMB_in"][:, :W]
    CONS_in = (sc_master["CONS_in"] if W is None else sc_master["CONS_in"][:, :W, :]).sum(axis=2) * scale_in     # aggregated MW, with scaling

    if use_actual:
        # Use actual realized data
        EAM_test = sc_master["EAM_actual"]
        IMB_test = sc_master["IMB_actual"]
        CONS_test = sc_master["CONS_actual"] * scale_out  # Apply scaling
        N_test_realizations = 1
    else:
        # Use out-of-sample scenarios
        EAM_test = sc_master["EAM_out"]
        IMB_test = sc_master["IMB_out"]
        CONS_test = sc_master["CONS_out"] * scale_out  # Apply scaling
        N_test_realizations = EAM_test.shape[1]

    n_omega = EAM_in.shape[1]
    N_charge = range(CONS_test.shape[-1]) if use_actual else range(CONS_test.shape[2])

    for i, value in enumerate(param_values):
        print(f"\n=== Sensitivity {param_name} = {value}  ({i+1}/{len(param_values)}) ===")

        base_args = base_args_template.copy()
        base_args.update(dict(
            **{param_name: value},
            EAM_scenarios=EAM_in,
            min_imbalance_scenarios=IMB_in,
            consumption_scenarios=CONS_in,
            N_charge=N_charge,
            n_omega=n_omega,
        ))

        if not use_actual:
            base_args.update(dict(
                EAM_scenarios_out=EAM_test,
                min_imbalance_scenarios_out=IMB_test,
                consumption_scenarios_out=CONS_test,
            ))

        run_data = []
        for test_idx in range(N_test_realizations):
            if use_actual:
                print(f"  Running on actual realized data")
                args = base_args.copy()
                args.update(dict(
                    EAM_real=EAM_test,
                    IMB_real=IMB_test,
                    consumption_real=CONS_test,
                ))
            else:
                print(f"  OOS realization {test_idx+1}/{N_test_realizations}")
                args = base_args.copy()
                args["w_real"] = test_idx

            res = run_fleet_model(**args)
            all_results.append(res)
            run_data.append(_collect_run_stats(res, CONS_test, test_idx if not use_actual else 0))

            if plot_individual:
                plot_results(
                    base_args["WINDOWS"], res,
                    EAM_in, CONS_in, EAM_test, IMB_in, IMB_test, CONS_test,
                    w_real=test_idx if not use_actual else None,
                    show_bids=True, show_profit=True,
                    show_aggregated=True, show_cumulative=True,
                )

        summaries.append(_aggregate(run_data, extra={param_name: value}))

    df_summary = pd.DataFrame(summaries)

    if show_summary:
        label_map = {
            "alpha": r"$\alpha$",
            "theta": r"$\theta$",
            "W": "In-sample scenarios $|\\Omega|$",
        }
        x_label = label_map.get(param_name, param_name)
        plot_sensitivity_summary(df_summary, param_name, x_label)

    return df_summary, all_results
# %%