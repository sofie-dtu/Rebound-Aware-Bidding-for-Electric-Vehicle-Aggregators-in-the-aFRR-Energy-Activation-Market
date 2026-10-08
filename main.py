#%% main.py
import numpy as np
import importlib
from src import scenarios
from src import scenario_generation, bidding_model, funcitons1, sensitivity

print("Loading sensitivity from:", sensitivity.__file__)
for m in [scenarios, sensitivity, bidding_model, funcitons1, scenario_generation]:
    importlib.reload(m)

from scenarios import scenarios_from_pipeline, plot_selected_scenarios
from src.bidding_model import run_fleet_model
from funcitons1 import plot_results
from sensitivity import  sensitivity_scenarios, run_sensitivity, run_sensitivity_multiday
from src.scenario_generation import run_pipeline

#%%
# -------------------------------
# Set parameters
# -------------------------------
W = 250     # number of in-sample scenarios
W_out = 5   # number of out-of-sample scenarios

WINDOWS=[
            list(range(0, 25)),
            list(range(25, 50)),
            list(range(50, 80)),
            list(range(80, 96)),
        ] 
zone = "DK1"    # Grid zone
fee = 0
alpha = 0.8  # Capacity cap (share of baseline offered)
w_real = 0        # OOS scenario index, used only by the OOS sweeps below
theta = 50        # Risk-aversion percentile for the imbalance-price floor
k_neighbors = 17 
# %%
# -------------------------------
# Target day & scenario generation
# -------------------------------
# 26/11
target_day = "2025-12-18"   # <-- choose the operating day here
k_neighbors = 17            # number of similar days (scenarios) to select
seed = 1                    # OOS pair selection seed
SAVE_TO_DISK = False        # also write .npy files (not needed for a single run)

# Consumption outlier handling: drop selected days with long zero stretches that
# would otherwise cap every chance-constrained bid at their worst quarter.
drop_outliers_cons = True
cons_zero_cap = 0.15        # drop a day if >15% of its quarters are zero

# Generate samples + actuals for the target day and build the scenario set
# entirely in memory (no .npy round-trip).
pipeline_out = run_pipeline(
    forecast_day=target_day,
    zone=zone,
    k_neighbors=k_neighbors,
    save_dir="../thesis_files",
    save_to_disk=SAVE_TO_DISK,
    make_plots=True,
    drop_outliers_cons=drop_outliers_cons,
    cons_zero_cap=cons_zero_cap,
)

sc_master = scenarios_from_pipeline(pipeline_out, seed=seed, W_out=W_out,
                                    max_zero_fraction=cons_zero_cap)

# Verify scenario counts
print(f"\nScenario summary:")
print(f"  In-sample:  {sc_master['n_in']} combinations")
print(f"  Out-of-sample: {sc_master['n_out']} combinations (requested: {W_out})")
print(f"  Using first W={W} in-sample scenarios for model")

assert sc_master['n_out'] == W_out, f"OOS mismatch: got {sc_master['n_out']}, expected {W_out}"
assert W <= sc_master['n_in'], f"W={W} exceeds available in-sample scenarios: {sc_master['n_in']}"

# -------------------------------
# Load and plot scenarios
# -------------------------------
# Extract scenario data
EAM_in  = sc_master["EAM_in"][:, :W]
IMB_in  = sc_master["IMB_in"][:, :W]
CONS_in = sc_master["CONS_in"][:, :W, :].sum(axis=2)

EAM_out = sc_master["EAM_out"]
IMB_out = sc_master["IMB_out"]
CONS_out = sc_master["CONS_out"]

# Target-day actual realization (set by cluster_v2 / scenarios.load_scenarios)
EAM_actual = sc_master["EAM_actual"]
IMB_actual = sc_master["IMB_actual"]
CONS_actual = sc_master["CONS_actual"]

N_charge = range(CONS_out.shape[2])

# Day-ahead (spot) price of the target day, collected like the other actuals
spot_price = sc_master["SPOT_actual"]


# Average fleet consumption over time and scenarios (MW)
# --- NORMALIZATION ---
P_avg_out = CONS_out.sum(axis=2).mean()
P_avg_in = CONS_in.mean()
scale_out = 1.0 / P_avg_out
scale_in = 1.0 / P_avg_in

CONS_in  *= scale_in
# since CONS_out is per charger, we scale per charger too
CONS_out *= scale_out
# scale the actual realization on the same per-charger basis as CONS_out
if CONS_actual is not None:
    CONS_actual = CONS_actual * scale_out

plot_selected_scenarios(EAM_in, IMB_in, CONS_in,
                        W=W,
                        EAM_actual=EAM_actual, IMB_actual=IMB_actual, CONS_actual=CONS_actual)
#%%
# -------------------------------
# Run main model
# -------------------------------

results = run_fleet_model(
    fee=fee,
    spot_price=spot_price,
    alpha=alpha,
    WINDOWS=WINDOWS,
    EAM_scenarios=EAM_in,
    min_imbalance_scenarios=IMB_in,
    consumption_scenarios=CONS_in,
    N_charge=N_charge,
    n_omega=W,
    theta=theta,
    EAM_real=EAM_actual,
    IMB_real=IMB_actual,
    consumption_real=CONS_actual,
    compute_profit=True)

# -------------------------------
# Plot results
# -------------------------------
# %%

#%%
plot_results(WINDOWS, results, EAM_in, CONS_in, EAM_out, IMB_in, IMB_out, CONS_out,
             EAM_real=EAM_actual, IMB_real=IMB_actual, CONS_real=CONS_actual,
             show_bids=True, show_profit=True, show_charger=True,
             show_aggregated=True, show_aggregated_per_second=True, show_cumulative=True,
             show_revenue_breakdown=True, charger_ids=[3, 12, 25, 94])

# total activated energy in percentage out of total consumption energy
total_activated_energy = np.sum(results["E_act_q"])
total_energy = np.sum(CONS_actual) / 4
activated_energy_pct = 100 * total_activated_energy / total_energy
print(f"Total activated energy: {total_activated_energy:,.1f} MWh "
      f"({activated_energy_pct:.2f}% of total consumption energy)")

# actovation time percentage
c_bid_q_resampled = np.repeat(results["p_bid_q"], 900)        
pct_activation_time = 100 * np.mean(np.logical_and(results["X_sec"]>0,c_bid_q_resampled) > 0)
print(f"Percentage of time activated: {pct_activation_time:.2f}%")

#%%
# -------------------------------
# Sensitivity over ALPHA only
# -------------------------------
alpha_values = [0.5, 0.75, 0.8, 0.9, 1.0]
run_sensitivity(
    param_name="alpha",
    param_values=alpha_values,
    sc_master=sc_master,
    base_args_template=dict(
        fee=fee,
        spot_price=spot_price,
        WINDOWS=WINDOWS,
        theta=theta,
        compute_profit=True,
    ),
    plot_individual=True,
    show_summary=True,
    scale_in=scale_in,
    scale_out=scale_out,
    W=W,
)

#%%
# -------------------------------
# Sensitivity over theta, averaged across 5 held-out days
# -------------------------------
# Build sc_master + normalisation factors for each day.
# Each day is solved independently (its own in-sample scenarios for planning,
# its own realized prices/consumption for evaluation); results are averaged.
theta_days = [
    "2025-12-14",
    "2025-12-15",
    "2025-12-16",
    "2025-12-17",
    "2025-12-18",
]

day_data = []
for td in theta_days:
    pipeline_out_d = run_pipeline(
        forecast_day=td,
        zone=zone,
        k_neighbors=k_neighbors,
        save_dir="../thesis_files",
        save_to_disk=SAVE_TO_DISK,
        make_plots=False,
        drop_outliers_cons=drop_outliers_cons,
        cons_zero_cap=cons_zero_cap,
    )
    sc_d = scenarios_from_pipeline(pipeline_out_d, seed=seed, W_out=W_out,
                                   max_zero_fraction=cons_zero_cap)
    P_avg_out_d = sc_d["CONS_out"].sum(axis=2).mean()
    P_avg_in_d  = sc_d["CONS_in"][:, :W, :].sum(axis=2).mean()
    day_data.append({
        "date":      td,
        "sc_master": sc_d,
        "scale_in":  1.0 / P_avg_in_d,
        "scale_out": 1.0 / P_avg_out_d,
    })

percentile_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]
run_sensitivity_multiday(
    param_name="theta",
    param_values=percentile_values,
    day_data=day_data,
    base_args_template=dict(
        fee=fee,
        alpha=alpha,
        WINDOWS=WINDOWS,
        compute_profit=True,
    ),
    plot_individual=False,
    show_summary=True,
    W=W,
)


# %%
# -------------------------------
# Sensitivity over number of in-sample scenarios
# -------------------------------

W_values = [10, 40, 60, 100, 150, 200, 240, 270]

summary_W, results_W = sensitivity_scenarios(
    W_values=W_values,
    sc_master=sc_master,
    base_args_template=dict(
        fee=fee,
        spot_price=spot_price,
        WINDOWS=WINDOWS,
        alpha=alpha,
        theta=theta,
        compute_profit=True,
    ),
    plot_individual=False,
    use_actual=True,
    scale_in=scale_in,
    scale_out=scale_out,
    W=None,
)



#%%
# =========================================
# Hyperparameter grid (2 values each)
# =========================================

alpha_list = [0.75, 1.0]
theta_list = [50, 75, 90]   # imbalance-price percentile

# =========================================
# Run grid search
# =========================================

from itertools import product
import time
import pandas as pd


results_summary = []

output_file = "hyperparameter_results.csv"

start_all = time.time()

for alpha, theta in product(alpha_list, theta_list):

    print("\n" + "="*60)
    print(f"Running alpha={alpha}, theta={theta}")
    print("="*60)

    try:
        results = run_fleet_model(
            fee=fee,
            spot_price=spot_price,
            alpha=alpha,
            WINDOWS=WINDOWS,
            EAM_scenarios=EAM_in,
            min_imbalance_scenarios=IMB_in,
            consumption_scenarios=CONS_in,
            EAM_scenarios_out=EAM_out,
            min_imbalance_scenarios_out=IMB_out,
            consumption_scenarios_out=CONS_out,
            N_charge=N_charge,
            n_omega=W,
            w_real=w_real,
            theta=theta,
            compute_profit=True,
        )

        # -----------------------------
        # KPIs
        # -----------------------------

        profit = results["profit_net_total"]
        print(profit)

        # % of quarters with positive bid
        pct_bid_time = 100 * np.mean(results["p_bid_q"] > 0)
        print(pct_bid_time)

        # % of time activated (second resolution)
        c_bid_q_resampled = np.repeat(results["p_bid_q"], 900)
        
        pct_activation_time = 100 * np.mean(np.logical_and(results["X_sec"]>0,c_bid_q_resampled) > 0)
        print(pct_activation_time)

        # total activated energy 
        total_activated_energy = np.sum(results["E_act_q"])
        print(total_activated_energy)

        total_energy = np.sum(CONS_out[:, w_real, :])/4 
        print(total_energy)

        activated_energy_pct = 100 * total_activated_energy / total_energy
        print(activated_energy_pct)

        # average backlog across flexibility windows
        avg_backlog = np.mean(results["backlog_end_window"])
        print(avg_backlog)

        runtime = results["runtime_min"]

        results_summary.append({
            "alpha": alpha,
            "theta": theta,
            "profit_eur": profit,
            "total_activated_energy_kwh": total_activated_energy,
            "avg_backlog_kwh": avg_backlog,
            "pct_bid_time": pct_bid_time,
            "pct_activation_time": pct_activation_time,
            "activated_energy_pct": activated_energy_pct,
            "total energy": total_energy,
            "runtime_min": runtime,
        })


        # -----------------------------
        # Save incrementally (SAFE)
        # -----------------------------

        df_tmp = pd.DataFrame(results_summary)
        df_tmp.to_csv(output_file, index=False)

        print(f"Finished | Profit = {profit:,.2f} EUR | "
              f"Bid time = {pct_bid_time:.1f}% | "
              f"Activation = {pct_activation_time:.1f}%")

    except Exception as e:
        print("ERROR during run:")
        print(e)

end_all = time.time()

print("\n" + "="*60)
print(f"ALL RUNS COMPLETED in {(end_all - start_all)/60:.1f} minutes")
print(f"Results saved to: {output_file}")
print("="*60)