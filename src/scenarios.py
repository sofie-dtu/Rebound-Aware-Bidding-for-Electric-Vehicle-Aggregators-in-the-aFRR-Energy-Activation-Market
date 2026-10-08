# %% scenarios.py
"""
Build in-sample / out-of-sample scenario sets from the similar-day sample banks
produced by cluster_v2.py, and load the target day's actual realized profiles.

Workflow:
    1. cluster_v2.py saves the sample banks and the target-day actuals.
    2. generate_fixed_oos_pairs(...) fixes a reproducible OOS subset.
    3. load_scenarios(...) returns the full in-sample master set, the fixed OOS
       set (kept for diagnostics), and the target-day actuals used as the
       realization in the bidding model.
"""

import os
import itertools
import random

import numpy as np
import matplotlib.pyplot as plt


# ===========================================================
# 1) Build scenario sets from sample banks
# ===========================================================
def _filter_consumption(consumption_samples, max_zero_fraction=0.25):
    """
    Drop per-box consumption scenarios with too many zero quarters.

    The zero fraction is measured on the AGGREGATED profile (sum over boxes),
    which is exactly what the model and the scenario plot use, so this is the
    single source of truth for which days are too zero-heavy. A day with long
    zero stretches would otherwise cap every chance-constrained bid at its worst
    quarter. The default 25% matches cons_zero_cap in cluster_v2.
    """
    summed = consumption_samples.sum(axis=2)
    zero_fraction = np.mean(summed == 0, axis=0)
    mask = zero_fraction <= max_zero_fraction
    n_dropped = int((~mask).sum())
    if n_dropped > 0:
        fr = [f"{f:.0%}" for f in zero_fraction[~mask]]
        print(f"  _filter_consumption dropped {n_dropped} zero-heavy scenario(s): {fr}")
    return consumption_samples[:, mask, :]


def make_oos_pairs(N_price, N_cons, seed, W_out):
    """Pick a reproducible set of out-of-sample (price, consumption) index pairs."""
    if W_out > N_price * N_cons:
        raise ValueError(f"W_out ({W_out}) exceeds total combinations ({N_price * N_cons})")
    all_combinations = list(itertools.product(range(N_price), range(N_cons)))
    rng = random.Random(seed)
    return rng.sample(all_combinations, W_out)


def _assemble(EAM_samples, imbalance_samples, consumption_samples,
              pairs_out, actuals):
    """Stack IS/OOS scenario matrices for the given OOS pairs and pack the dict."""
    N_price = EAM_samples.shape[1]
    N_cons = consumption_samples.shape[1]
    all_pairs = list(itertools.product(range(N_price), range(N_cons)))
    insample_pairs = sorted(set(all_pairs) - set(pairs_out))

    print(f"Total valid in-sample pairs: {len(insample_pairs)}")
    print(f"Fixed OOS pairs: {len(pairs_out)}")

    def stack(pairs):
        EAM = np.stack([EAM_samples[:, i] for (i, j) in pairs], axis=1)
        IMB = np.stack([imbalance_samples[:, i] for (i, j) in pairs], axis=1)
        CONS = np.stack([consumption_samples[:, j] for (i, j) in pairs], axis=1)
        return EAM, IMB, CONS

    EAM_in, IMB_in, CONS_in = stack(insample_pairs)
    EAM_out, IMB_out, CONS_out = stack(pairs_out)

    actuals = actuals or {}
    return {
        "EAM_in": EAM_in,
        "IMB_in": IMB_in,
        "CONS_in": CONS_in,
        "EAM_out": EAM_out,
        "IMB_out": IMB_out,
        "CONS_out": CONS_out,
        "EAM_actual": actuals.get("EAM_actual") or actuals.get("EAM"),
        "IMB_actual": actuals.get("IMB_actual") or actuals.get("IMB"),
        "CONS_actual": actuals.get("CONS_actual") or actuals.get("CONS"),
        "SPOT_actual": actuals.get("SPOT_actual") or actuals.get("SPOT"),
        "pairs_in": insample_pairs,
        "pairs_out": pairs_out,
        "n_in": len(insample_pairs),
        "n_out": len(pairs_out),
    }


def build_scenarios(EAM_samples, imbalance_samples, consumption_samples,
                    seed, W_out, actuals=None, max_zero_fraction=0.25):
    """
    Assemble the in-sample / out-of-sample scenario set from sample banks held
    in memory (no disk I/O).

    Parameters
    ----------
    EAM_samples : np.ndarray, shape (n_steps, N_price)
    imbalance_samples : np.ndarray, shape (96, N_price)
    consumption_samples : np.ndarray, shape (96, N_cons, n_box)
        Per-box consumption bank (already scaled). Scenarios whose aggregated
        profile is more than max_zero_fraction zeros are filtered out here.
    seed, W_out : int
        Reproducible OOS pair selection.
    actuals : dict or None
        Optional target-day actuals (keys SPOT_actual/EAM_actual/IMB_actual/
        CONS_actual) passed straight through into the returned dict.
    max_zero_fraction : float
        Drop consumption scenarios with a larger fraction of zero quarters.

    Returns
    -------
    dict identical in structure to load_scenarios(...).
    """
    consumption_samples = _filter_consumption(consumption_samples, max_zero_fraction)
    pairs_out = make_oos_pairs(EAM_samples.shape[1], consumption_samples.shape[1], seed, W_out)
    return _assemble(EAM_samples, imbalance_samples, consumption_samples, pairs_out, actuals)


def scenarios_from_pipeline(pipeline_out, seed, W_out, max_zero_fraction=0.25):
    """
    Build the scenario set directly from cluster_v2.run_clustering_pipeline(...)
    output, entirely in memory.
    """
    return build_scenarios(
        EAM_samples=pipeline_out["EAM_samples"],
        imbalance_samples=pipeline_out["imbalance_samples"],
        consumption_samples=pipeline_out["consumption_per_box_samples"],
        seed=seed,
        W_out=W_out,
        actuals=pipeline_out.get("actuals"),
        max_zero_fraction=max_zero_fraction,
    )


# ===========================================================
# 1b) Disk-based variants (kept for standalone use)
# ===========================================================
def generate_fixed_oos_pairs(zone, seed, W_out):
    """
    Fix a reproducible set of OOS index pairs from the on-disk banks and save
    them. Kept for the file-based workflow; the in-memory path uses
    build_scenarios / scenarios_from_pipeline instead.
    """
    EAM_samples = np.load(f"../thesis_files/EAM_price_samples_{zone}.npy")
    consumption_samples = _filter_consumption(
        np.load("../thesis_files/consumption_per_box_sample.npy"))

    pairs_out = make_oos_pairs(EAM_samples.shape[1], consumption_samples.shape[1], seed, W_out)
    output_file = f"fixed_oos_pairs_{zone}.npy"
    np.save(output_file, np.array(pairs_out, dtype=int))
    print(f"Saved {W_out} fixed OOS pairs to {os.path.abspath(output_file)} (seed={seed})")
    return pairs_out


def load_target_day_actuals(zone):
    """
    Load the target day's actual realized day-ahead (spot) price, EAM price,
    imbalance price, and per-charger consumption from disk. Returns None if the
    files are absent.
    """
    try:
        SPOT_actual = np.load(f"../thesis_files/spot_price_forecast_day_{zone}.npy")
        EAM_actual = np.load(f"../thesis_files/EAM_actual_{zone}.npy")
        IMB_actual = np.load(f"../thesis_files/Imbalance_actual_{zone}.npy")
        CONS_actual = np.load("../thesis_files/consumption_actual_per_box.npy")
    except FileNotFoundError:
        print("Warning: target-day actuals not found; run cluster_v2 first.")
        return None

    print(f"Loaded target-day actuals: SPOT {SPOT_actual.shape}, EAM {EAM_actual.shape}, "
          f"IMB {IMB_actual.shape}, CONS {CONS_actual.shape}")
    return {"SPOT_actual": SPOT_actual, "EAM_actual": EAM_actual,
            "IMB_actual": IMB_actual, "CONS_actual": CONS_actual}


def load_scenarios(zone):
    """
    Disk-based variant of build_scenarios: load the sample banks, the fixed OOS
    pairs, and the target-day actuals from .npy, then assemble the scenario set.
    Requires generate_fixed_oos_pairs() to have been called first.
    """
    EAM_samples = np.load(f"../thesis_files/EAM_price_samples_{zone}.npy")
    imbalance_samples = np.load(f"../thesis_files/imbalance_price_samples_{zone}.npy")
    consumption_samples = _filter_consumption(
        np.load("../thesis_files/consumption_per_box_sample.npy"))

    try:
        pairs_out_fixed = np.load(f"fixed_oos_pairs_{zone}.npy", allow_pickle=True)
        pairs_out_fixed = [tuple(map(int, x)) for x in pairs_out_fixed]
        print(f"Loaded {len(pairs_out_fixed)} fixed OOS pairs from fixed_oos_pairs_{zone}.npy")
    except FileNotFoundError:
        print(f"Warning: fixed_oos_pairs_{zone}.npy not found. Call generate_fixed_oos_pairs() first!")
        pairs_out_fixed = []

    actuals = load_target_day_actuals(zone) or {}
    return _assemble(EAM_samples, imbalance_samples, consumption_samples,
                     pairs_out_fixed, actuals)


# ===========================================================
# 2) Plot in-sample and out-of-sample scenarios
# ===========================================================
def plot_selected_scenarios(EAM_in, IMB_in, CONS_in,
                            W,
                            EAM_actual=None, IMB_actual=None, CONS_actual=None):
    """Overlay in-sample scenarios and (optionally) the actual realization."""

    def plot_set(data_in, title, ylabel, actual=None):
        x = np.arange(data_in.shape[0]) / 4

        plt.figure(figsize=(14, 6))
        for i in range(W):
            plt.plot(x, data_in[:, i], alpha=0.4, lw=1)
        if actual is not None:
            plt.plot(x, actual, color="black", lw=2, label="Actual")
            plt.legend()
        plt.title(f"{title} - In-sample ({W} scenarios)")
        plt.xlabel("Hour")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    plot_set(EAM_in, "EAM Prices", "\u20ac/MWh", actual=EAM_actual)
    plot_set(IMB_in, "Imbalance Prices", "\u20ac/MWh", actual=IMB_actual)

    # CONS is per-charger; aggregate for plotting.
    CONS_in_agg = CONS_in.sum(axis=2) if CONS_in.ndim == 3 else CONS_in
    CONS_actual_agg = CONS_actual.sum(axis=1) if (CONS_actual is not None and CONS_actual.ndim == 2) else CONS_actual
    plot_set(CONS_in_agg, "Consumption", "MW", actual=CONS_actual_agg)
# %%