#%%
"""
Scenario generation by similar-day selection (k-nearest analogues).

For a given forecast day, this module:
1. Loads historical EAM, imbalance, consumption, spot, and weather data.
2. Builds daily 96-step (15-min) profiles for each series.
3. Selects the k historical days most similar to the forecast day on its known
   day-ahead drivers -- spot price + wind for the price series, temperature for
   the consumption series, plus calendar features in both.
4. Extracts the raw profiles of the selected days as scenario matrices.

The selected days act as analogue scenarios for the unknown target series
(EAM, imbalance, realized load). Visualization functions are in
visualization.py; matrix extraction is in utils.py.
"""

import os
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from visualization import (
    plot_forecast_features,
    plot_raw_days_multi,
    plot_all_days_on_ax,
)
from src.utils import extract_raw_scenarios_matrix

# =========================================================
# Settings
# =========================================================
TIME_START = "2025-10-01T00:00"
TIME_END = "2026-04-29T00:00"
ZONE = "DK1"
N_STEPS = 96            # 15-min steps per day

# =========================================================
# Data loading
# =========================================================
def get_data(zone):
    """Load historical EAM, imbalance, consumption, spot, and weather data."""
    cwd = os.getcwd()
    base = os.path.abspath(os.path.join(cwd, "data"))

    EAM = pd.read_csv(os.path.join(base, f"EAM/EAM_clean_{zone}.csv"))
    IMB = pd.read_csv(os.path.join(base, f"Imbalance/Imbalance_clean_{zone}.csv"))
    CONS = pd.read_csv(os.path.join(base, "Consumption/Consumption_agg_15min.csv"))
    CONS_per_box = pd.read_csv(os.path.join(base, "Consumption/Consumption_perBox_15min.csv"), sep=",")
    SPOT = pd.read_csv(os.path.join(base, f"SpotPrice/SpotPrice_clean_{zone}.csv"))

    W = pd.read_csv(os.path.join(base, "Weather/Weather_clean.csv"))
    W["timestamp"] = pd.to_datetime(W["timestamp"], utc=True)
    W = W.set_index("timestamp").sort_index()
    W.index = W.index.tz_convert(None)

    start = pd.to_datetime(TIME_START)
    end = pd.to_datetime(TIME_END)
    W = W[(W.index >= start) & (W.index < end)]
    W_15 = W.resample("15min").interpolate("time")

    return EAM, IMB, CONS, SPOT, W_15, CONS_per_box

def get_similar_day_stats(similar_days, profiles_df):
    """Summary statistics (date range, weekday/month mix) for selected days."""
    similar_days = pd.to_datetime(similar_days)
    subset = profiles_df.loc[similar_days]

    return {
        "n_scenarios": len(similar_days),
        "date_range": (similar_days.min().date(), similar_days.max().date()),
        "dow_distribution": subset["dow"].value_counts().to_dict(),
        "month_distribution": subset["month"].value_counts().to_dict(),
        "weekend_share": subset["is_weekend"].mean(),
        "most_common_dow": subset["dow"].mode()[0] if len(subset) > 0 else None,
    }


def parse_datetime(df, column="timestamp"):
    """Parse a datetime column and set it as a sorted index."""
    df[column] = pd.to_datetime(df[column])
    return df.set_index(column).sort_index()


def make_daily_profiles(df, value_col, prefix):
    """Reshape a 15-min series into one row per day with N_STEPS columns."""
    df = df.sort_index()
    grouped = df[value_col].groupby(df.index.normalize()).apply(list)
    daily = pd.DataFrame(grouped.tolist(), index=grouped.index)
    daily.columns = [f"{prefix}t{i:02d}" for i in range(daily.shape[1])]
    return daily


def load_and_prepare_data():
    """Load all series, parse datetimes, and clip to the configured window."""
    EAM, IMB, CONS, SPOT, weather_15, CONS_per_box = get_data(ZONE)

    EAM = parse_datetime(EAM, "timestamp")
    IMB = parse_datetime(IMB, "timestamp")
    CONS = parse_datetime(CONS, "timestamp")
    SPOT = parse_datetime(SPOT, "timestamp")
    CONS_per_box = parse_datetime(CONS_per_box, "timestamp")

    t_min = pd.to_datetime(TIME_START)
    t_max = pd.to_datetime(TIME_END) - pd.Timedelta(minutes=15)

    CONS = CONS.loc[t_min:t_max]
    SPOT = SPOT.loc[t_min:t_max]
    IMB = IMB.loc[t_min:t_max]
    EAM = EAM.loc[t_min:t_max]
    weather_15 = weather_15.loc[t_min:t_max]

    return {
        "EAM": EAM,
        "imbalance": IMB,
        "charger": CONS,
        "spot": SPOT,
        "weather": weather_15,
        "charger_per_box": CONS_per_box,
        "t_min": t_min,
    }


# =========================================================
# Similar-day selection
# =========================================================
def _candidate_mask(profiles, forecast_day, match_daytype=True, season_window=None):
    """
    Boolean mask over `profiles` restricting which historical days may be
    selected as analogues for `forecast_day`.

    match_daytype : bool
        If True, weekdays only match weekdays and weekends only match weekends.
        This is a hard gate -- load and price shapes differ markedly between the
        two, so they should never be mixed regardless of feature scaling.
    season_window : int or None
        If given, only days within +/- this many days-of-year of the forecast
        day are eligible, wrapping around the year. Restricts analogues to the
        same season.
    """
    forecast_day = pd.Timestamp(forecast_day)
    mask = pd.Series(True, index=profiles.index)

    if match_daytype:
        fc_weekend = forecast_day.weekday() >= 5
        mask &= (profiles.index.dayofweek >= 5) == fc_weekend

    if season_window is not None:
        doy = profiles.index.dayofyear
        fc_doy = forecast_day.dayofyear
        diff = np.abs(doy - fc_doy)
        circ = np.minimum(diff, 365 - diff)   # wrap around Dec/Jan
        mask &= circ <= season_window

    return mask.values


def _drop_zero_heavy_days(selected_days, cons_profiles, outlier_n_mad=3.0,
                          abs_zero_cap=0.25, min_keep=3):
    """
    From a set of selected consumption days, drop those with too many zero
    quarters. A day is dropped if its zero fraction exceeds either:
      * an absolute cap (abs_zero_cap, default 25% of the day), or
      * a robust group threshold, median + outlier_n_mad * MAD.

    This targets the model failure where one day with long zero stretches caps
    every chance-constrained bid at its worst quarter. Driver-space KNN can miss
    it because the day may still be a good temperature/calendar analogue; this
    check looks at the consumption profile itself.

    The absolute cap is the primary guard: when the group is tight (MAD near 0),
    the spread-based threshold alone can be fooled by a single extreme value
    inflating any variance estimate, so a fixed cap reliably catches a clearly
    broken day. Always keeps at least `min_keep` days (those with fewest zeros).
    """
    cols = [c for c in cons_profiles.columns if c.startswith("Cons_")]
    sub = cons_profiles.loc[selected_days, cols]
    zero_frac = (sub.values == 0).mean(axis=1)

    med = np.median(zero_frac)
    mad = np.median(np.abs(zero_frac - med))
    group_threshold = med + outlier_n_mad * mad if mad > 1e-12 else np.inf

    # Drop if above the absolute cap OR an outlier within a (non-degenerate) group.
    drop = (zero_frac > abs_zero_cap) | (zero_frac > group_threshold)
    keep = ~drop

    if keep.sum() < min_keep:
        # Too aggressive for this group; keep the least-zero days instead.
        order = np.argsort(zero_frac)
        keep = np.zeros(len(zero_frac), dtype=bool)
        keep[order[:min_keep]] = True

    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        dropped = [str(d.date()) for d in selected_days[~keep]]
        fr = [f"{f:.0%}" for f in zero_frac[~keep]]
        print(f"  dropped {n_dropped} zero-heavy consumption day(s): "
              f"{list(zip(dropped, fr))}")

    return selected_days[keep]


def _knn_days(profiles, feature_cols, query, k_neighbors,
              candidate_mask=None, drop_outliers=False,
              outlier_buffer=5, outlier_n_mad=3.0):
    """
    Return the k historical days whose `feature_cols` are nearest to `query`.

    Features are robustly scaled (median/IQR) so price levels and calendar
    features contribute on a comparable scale and outliers do not dominate.

    Parameters
    ----------
    profiles : pd.DataFrame
        Daily profiles indexed by date; must contain feature_cols.
    feature_cols : list[str]
        Columns used for the distance (known drivers only).
    query : dict
        Forecast-day values for feature_cols.
    k_neighbors : int
        Number of neighbours to return.
    candidate_mask : np.ndarray[bool] or None
        Optional mask over `profiles` rows limiting eligible analogue days
        (e.g. same weekday/weekend type). Scaling is still fit on all days so
        the distance metric is unaffected by the restriction.
    drop_outliers : bool
        If True, over-select k+outlier_buffer nearest days, drop any that sit
        far from the selected group (relative to the group's own spread), then
        keep the k closest survivors. This removes a day that KNN admitted but
        which is an outlier versus the rest of the selection -- e.g. a day with
        long zero stretches that would otherwise cap every chance-constrained
        bid at its worst quarter.
    outlier_buffer : int
        Extra candidates pulled before outlier removal.
    outlier_n_mad : float
        A day is an outlier if its distance to the group median exceeds
        median + outlier_n_mad * MAD (median absolute deviation). Larger keeps
        more days; smaller is stricter.

    Returns
    -------
    days : pd.DatetimeIndex
    X : np.ndarray
        Scaled feature matrix for all historical days (for plotting/diagnostics).
    """
    scaler = RobustScaler()
    X = scaler.fit_transform(profiles[feature_cols])
    x_query = scaler.transform(pd.DataFrame([query])[feature_cols])

    if candidate_mask is None:
        candidate_mask = np.ones(len(profiles), dtype=bool)

    X_cand = X[candidate_mask]
    cand_days = profiles.index[candidate_mask]

    if not drop_outliers:
        k = min(k_neighbors, len(cand_days))
        if k < k_neighbors:
            print(f"  [warn] only {len(cand_days)} candidate days available "
                  f"(requested k={k_neighbors})")
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_cand)
        _, idx = nn.kneighbors(x_query)
        return cand_days[idx[0]], X

    # --- Over-select, then drop group outliers ---
    k_over = min(k_neighbors + outlier_buffer, len(cand_days))
    nn = NearestNeighbors(n_neighbors=k_over)
    nn.fit(X_cand)
    _, idx = nn.kneighbors(x_query)
    sel_idx = idx[0]
    X_sel = X_cand[sel_idx]

    # Distance of each selected day to the robust centre of the selected group.
    centre = np.median(X_sel, axis=0)
    dist = np.linalg.norm(X_sel - centre, axis=1)
    med = np.median(dist)
    mad = np.median(np.abs(dist - med))
    # Fall back to std if MAD is degenerate (all distances equal).
    spread = mad if mad > 1e-12 else dist.std()
    threshold = med + outlier_n_mad * spread

    keep = dist <= threshold
    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        dropped_days = [str(d.date()) for d in cand_days[sel_idx[~keep]]]
        print(f"  dropped {n_dropped} outlier day(s) from selection: {dropped_days}")

    # Keep survivors ordered by nearness to the query, take the k closest.
    survivors = sel_idx[keep]
    k = min(k_neighbors, len(survivors))
    if k < k_neighbors:
        print(f"  [warn] only {len(survivors)} days remain after outlier removal "
              f"(requested k={k_neighbors})")
    chosen = survivors[:k]
    return cand_days[chosen], X


def get_similar_days_for_forecast(
    t_min,
    forecast_day,
    df_charger,
    spot_price_DK1,
    imbalance_DK1,
    EAM_price_DK1,
    df_weather_15,
    k_neighbors=15,
    match_daytype_price=True,
    match_daytype_cons=True,
    season_window_price=None,
    season_window_cons=None,
    drop_outliers_price=False,
    drop_outliers_cons=True,
    outlier_n_mad=3.0,
    cons_zero_cap=0.25,
):
    """
    Select the k historical days most similar to forecast_day, using only the
    drivers known ahead of time. Two independent selections are made:

      * price days   -- nearest on the day-ahead spot price, wind, and calendar
                        features (weekday / weekend / month);
      * consumption days -- nearest on temperature and calendar features.

    Candidate days are first hard-filtered to the same day type (weekday vs
    weekend) as the forecast day, because price and especially load shapes
    differ sharply between the two and calendar features alone are too weak to
    keep them apart against 96-dimensional driver profiles. An optional season
    window further restricts candidates to a band of days-of-year around the
    forecast day.

    The unknown target series (EAM, imbalance, realized load) are deliberately
    excluded from the distance, since they are what the scenarios predict.

    Parameters
    ----------
    match_daytype_price, match_daytype_cons : bool
        Restrict candidates to the same weekday/weekend type as the forecast
        day, for the price and consumption selections respectively.
    season_window_price, season_window_cons : int or None
        If set, restrict candidates to +/- this many days-of-year around the
        forecast day (wraps around the year-end).

    Returns
    -------
    similar_price_days, similar_cons_days : pd.DatetimeIndex
    price_profiles, cons_profiles : pd.DataFrame (full daily profiles)
    X_price, X_cons : np.ndarray (scaled driver matrices used for KNN)
    spot_vals, temp_vals, wind_vals : np.ndarray (forecast-day inputs)
    """
    forecast_day = pd.to_datetime(forecast_day)
    train_end = forecast_day - pd.Timedelta(days=1)

    # --- 0) Training window ---
    df_charger_w = df_charger.loc[t_min:train_end]
    spot_price_w = spot_price_DK1.loc[t_min:train_end]
    imbalance_w = imbalance_DK1.loc[t_min:train_end]
    EAM_price_w = EAM_price_DK1.loc[t_min:train_end]
    weather_w = df_weather_15.loc[t_min:train_end]

    # --- 1) Daily 15-min profiles ---
    spot_profiles = make_daily_profiles(spot_price_w.resample("15min").ffill(), "DayAheadPriceEUR", "Spot_")
    imb_profiles = make_daily_profiles(imbalance_w.resample("15min").mean(), "ImbalancePriceEUR", "Imb_")
    eam_profiles = make_daily_profiles(EAM_price_w.resample("15min").mean(), "aFRR_ActivatedEUR", "EAM_")
    cons_profiles = make_daily_profiles(df_charger_w.resample("15min").mean(), "power", "Cons_")
    temp_profiles = make_daily_profiles(weather_w, "temperature_2m", "Temp_")
    wind_profiles = make_daily_profiles(weather_w, "wind_speed_10m", "Wind_")

    # --- 2) Forecast-day inputs (known day-ahead) ---
    forecast_day_ts = pd.Timestamp(forecast_day)
    forecast_start = forecast_day_ts.normalize()
    forecast_end = (forecast_day_ts + pd.Timedelta(days=1)).normalize()
    forecast_index = pd.date_range(
        start=forecast_day_ts,
        end=forecast_day_ts + pd.Timedelta(days=1) - pd.Timedelta(minutes=15),
        freq="15min",
    )
    spot_fc = spot_price_DK1.loc[forecast_start:forecast_end].resample("15min").ffill()
    spot_vals = spot_fc["DayAheadPriceEUR"].reindex(forecast_index, method="ffill").values

    weather_fc = df_weather_15.loc[forecast_start:forecast_end].resample("15min").interpolate("time")
    temp_vals = weather_fc["temperature_2m"].reindex(forecast_index, method="nearest").values
    wind_vals = weather_fc["wind_speed_10m"].reindex(forecast_index, method="nearest").values

    dow = forecast_day.weekday()
    is_weekend = int(dow >= 5)
    month = forecast_day.month
    print(f"Forecast day: {forecast_day.date()} (dow={dow}, weekend={is_weekend}, month={month})")

    # --- 3) PRICE selection: KNN on spot + wind + calendar ---
    # Keep the full price profiles (for extracting EAM/imbalance scenarios later)
    # but build the KNN distance only from known drivers.
    price_profiles = pd.concat([spot_profiles, imb_profiles, eam_profiles, wind_profiles], axis=1).dropna()
    price_profiles["dow"] = price_profiles.index.dayofweek
    price_profiles["is_weekend"] = (price_profiles["dow"] >= 5).astype(int)
    price_profiles["month"] = price_profiles.index.month

    price_features = [c for c in price_profiles.columns
                      if c.startswith(("Spot_", "Wind_"))] + ["dow", "is_weekend", "month"]

    price_query = {f"Spot_t{i:02d}": spot_vals[i] for i in range(N_STEPS)}
    price_query.update({f"Wind_t{i:02d}": wind_vals[i] for i in range(N_STEPS)})
    price_query.update(dow=dow, is_weekend=is_weekend, month=month)

    price_mask = _candidate_mask(price_profiles, forecast_day,
                                 match_daytype=match_daytype_price,
                                 season_window=season_window_price)
    similar_price_days, X_price = _knn_days(price_profiles, price_features, price_query,
                                            k_neighbors, candidate_mask=price_mask,
                                            drop_outliers=drop_outliers_price,
                                            outlier_n_mad=outlier_n_mad)
    print(f"  price: {price_mask.sum()} candidate days "
          f"({'same day-type' if match_daytype_price else 'all day-types'})")

    # --- 4) CONSUMPTION selection: KNN on temperature + calendar ---
    cons_profiles = pd.concat([cons_profiles, temp_profiles], axis=1).dropna()
    cons_profiles["dow"] = cons_profiles.index.dayofweek
    cons_profiles["is_weekend"] = (cons_profiles["dow"] >= 5).astype(int)
    cons_profiles["month"] = cons_profiles.index.month

    cons_features = [c for c in cons_profiles.columns
                     if c.startswith("Temp_")] + ["dow", "is_weekend", "month"]

    cons_query = {f"Temp_t{i:02d}": temp_vals[i] for i in range(N_STEPS)}
    cons_query.update(dow=dow, is_weekend=is_weekend, month=month)

    cons_mask = _candidate_mask(cons_profiles, forecast_day,
                                match_daytype=match_daytype_cons,
                                season_window=season_window_cons)
    similar_cons_days, X_cons = _knn_days(cons_profiles, cons_features, cons_query,
                                          k_neighbors, candidate_mask=cons_mask,
                                          drop_outliers=drop_outliers_cons,
                                          outlier_n_mad=outlier_n_mad)

    # Targeted guard for the model: drop any selected day whose consumption
    # profile has an outlying fraction of zero quarters. Such a day caps every
    # chance-constrained bid at its (near-zero) worst quarter even if it was a
    # fine temperature analogue.
    if drop_outliers_cons:
        similar_cons_days = _drop_zero_heavy_days(
            similar_cons_days, cons_profiles, outlier_n_mad=outlier_n_mad,
            abs_zero_cap=cons_zero_cap)

    print(f"  consumption: {cons_mask.sum()} candidate days "
          f"({'same day-type' if match_daytype_cons else 'all day-types'})")

    return (similar_price_days, similar_cons_days, price_profiles, cons_profiles,
            X_price, X_cons, spot_vals, temp_vals, wind_vals)


def extract_target_day_actuals(
    forecast_day,
    EAM_price_DK1,
    imbalance_DK1,
    df_charger_per_box,
    spot_price_DK1,
    save_dir=None,
    zone=ZONE,
):
    """
    Extract the *actual* realized profiles of the target (forecast) day.

    These are the true day-ahead (spot) price, EAM price, imbalance price, and
    per-charger consumption observed on the target day. The day-ahead price is
    known in advance and used directly in the model objective; the others are
    the realization that the bids -- optimized on the in-sample scenarios -- are
    settled against, making the evaluation a genuine out-of-sample test on the
    real day.

    The same extractor as the scenario sample banks is used (with the target
    day as a single "selected day"), so the returned arrays have exactly the
    same per-step resolution and per-charger layout as EAM_out / CONS_out and
    drop straight into the realization step. The spot series is resampled to a
    complete 15-min grid (forward-filled) before extraction, so day-ahead price
    steps are carried across the quarter instead of being zero-filled.

    Parameters
    ----------
    forecast_day : str or Timestamp
        The target operating day.
    EAM_price_DK1, imbalance_DK1, df_charger_per_box, spot_price_DK1 : pd.DataFrame
        Historical EAM price, imbalance price, per-charger consumption, and
        day-ahead (spot) price.
    save_dir : str or None
        If given, the actuals are also saved as .npy files there.
    zone : str
        Zone tag used in saved filenames.

    Returns
    -------
    dict with keys 'SPOT', 'EAM', 'IMB', 'CONS' (numpy arrays) and 'date'.
        SPOT : shape (96,)                 actual day-ahead price profile
        EAM  : shape (n_steps_eam,)        actual EAM price profile
        IMB  : shape (96,)                 actual imbalance price profile
        CONS : shape (96, n_chargers)      actual per-charger consumption [MW]
    """
    forecast_day = pd.to_datetime(forecast_day)
    days = pd.DatetimeIndex([forecast_day.normalize()])

    # Resample spot to a complete 15-min grid so the extractor sees a value in
    # every quarter (day-ahead price is piecewise-constant across the hour).
    spot_15 = spot_price_DK1.resample("15min").ffill()

    spot = extract_raw_scenarios_matrix(spot_15, days, value_col="DayAheadPriceEUR", return_format="dict")
    eam = extract_raw_scenarios_matrix(EAM_price_DK1, days, value_col="aFRR_ActivatedEUR", return_format="dict")
    imb = extract_raw_scenarios_matrix(imbalance_DK1, days, value_col="ImbalancePriceEUR", return_format="dict")
    cons = extract_raw_scenarios_matrix(df_charger_per_box, days, value_col="power", box_col="chargeBoxId", return_format="dict")

    # Drop the singleton scenario axis: (steps, 1) -> (steps,) and
    # (96, 1, n_box) -> (96, n_box). Consumption is scaled like the sample bank.
    SPOT_actual = np.asarray(spot["data"])[:, 0]
    EAM_actual = np.asarray(eam["data"])[:, 0]
    IMB_actual = np.asarray(imb["data"])[:, 0]
    CONS_actual = np.asarray(cons["data"])[:, 0, :] / 1000.0

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"spot_price_forecast_day_{zone}.npy"), SPOT_actual)
        np.save(os.path.join(save_dir, f"EAM_actual_{zone}.npy"), EAM_actual)
        np.save(os.path.join(save_dir, f"Imbalance_actual_{zone}.npy"), IMB_actual)
        np.save(os.path.join(save_dir, f"consumption_actual_per_box.npy"), CONS_actual)
        print(f"Saved target-day actuals for {forecast_day.date()} to {save_dir}")

    print(f"Target-day actuals: SPOT {SPOT_actual.shape}, EAM {EAM_actual.shape}, "
          f"IMB {IMB_actual.shape}, CONS {CONS_actual.shape}")
    return {"SPOT": SPOT_actual, "EAM": EAM_actual, "IMB": IMB_actual,
            "CONS": CONS_actual, "date": forecast_day.normalize()}


#%%
# =========================================================
# Pipeline entry point
# =========================================================
def run_pipeline(
    forecast_day,
    zone=ZONE,
    k_neighbors=17,
    save_dir="../thesis_files",
    save_to_disk=False,
    make_plots=True,
    drop_outliers_cons=True,
    outlier_n_mad=3.0,
    cons_zero_cap=0.25,
):
    """
    Full scenario-generation pipeline for one target day.

    Loads data, selects similar days for the target day, builds the
    price/consumption sample banks and the target-day actuals (spot, EAM,
    imbalance, per-charger consumption) in memory, optionally persists them to
    disk, and optionally draws the overview/selection figures.

    Parameters
    ----------
    forecast_day : str or Timestamp
        Target operating day.
    zone : str
        Market zone (used in filenames).
    k_neighbors : int
        Number of similar days (scenarios) to select.
    save_dir : str
        Directory for the saved .npy files (only used if save_to_disk).
    save_to_disk : bool
        If True, also write the sample banks and actuals to .npy. The in-memory
        return value is identical either way; saving is only for reuse across
        separate runs.
    make_plots : bool
        If True, draw the diagnostic figures.

    Returns
    -------
    dict with the in-memory sample banks, actuals, selected days, and forecast
    inputs. Pass it to scenarios.build_scenarios(...) to assemble the IS/OOS set
    without touching disk.
    """
    data = load_and_prepare_data()
    EAM_price_DK1 = data["EAM"]
    imbalance_DK1 = data["imbalance"]
    df_charger = data["charger"]
    spot_price_DK1 = data["spot"]
    df_weather_15 = data["weather"]
    df_charger_per_box = data["charger_per_box"]
    t_min = data["t_min"]

    (price_days, cons_days, price_prof, cons_prof,
     Xp, Xc, spot_vals, temp_vals, wind_vals) = get_similar_days_for_forecast(
        t_min=t_min,
        forecast_day=forecast_day,
        df_charger=df_charger,
        spot_price_DK1=spot_price_DK1,
        EAM_price_DK1=EAM_price_DK1,
        imbalance_DK1=imbalance_DK1,
        df_weather_15=df_weather_15,
        k_neighbors=k_neighbors,
        drop_outliers_cons=drop_outliers_cons,
        outlier_n_mad=outlier_n_mad,
        cons_zero_cap=cons_zero_cap,
    )

    if make_plots:
        # --- Overview of all historical days ---
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex=True,
                                 gridspec_kw={"hspace": 0.15, "wspace": 0.25})
        plot_all_days_on_ax(axes[0, 0], EAM_price_DK1, "aFRR_ActivatedEUR", r"$\lambda^{\text{EAM}}$ [\u20ac/MWh]")
        plot_all_days_on_ax(axes[0, 1], imbalance_DK1, "ImbalancePriceEUR", r"$\lambda^{\text{imb}}$ [\u20ac/MWh]")
        plot_all_days_on_ax(axes[1, 0], df_charger, "power", "Consumption [MW]")
        axes[1, 0].set_ylabel(r"$C^{\text{base}}$ [MW]")
        axes[1, 0].set_yticks([])
        axes[1, 0].spines["left"].set_visible(False)
        plot_all_days_on_ax(axes[1, 1], spot_price_DK1, "DayAheadPriceEUR", r"$\lambda^{\text{spot}}$ [\u20ac/MWh]")
        plot_all_days_on_ax(axes[2, 0], df_weather_15, "temperature_2m", "Temperature [\u00b0C]")
        plot_all_days_on_ax(axes[2, 1], df_weather_15, "wind_speed_10m", "Wind speed [m/s]")
        for ax in axes[:-1, :].flat:
            ax.tick_params(labelbottom=False)
        for ax in axes[-1, :]:
            ax.set_xlabel("Hour of day")
        plt.tight_layout()
        fig.savefig("fig_all_days_3x2.pdf", bbox_inches="tight")
        plt.show()

        # --- Selected similar days ---
        plot_raw_days_multi(
            panels=[
                {"df": EAM_price_DK1, "value_col": "aFRR_ActivatedEUR", "similar_days": price_days,
                 "title": "aFRR EAM prices", "ylabel": r"$\lambda^{\mathrm{EAM}}$ [\u20ac/MWh]"},
                {"df": imbalance_DK1, "value_col": "ImbalancePriceEUR", "similar_days": price_days,
                 "title": "Imbalance prices", "ylabel": r"$\lambda^{\mathrm{imb}}$ [\u20ac/MWh]"},
                {"df": df_charger, "value_col": "power", "similar_days": cons_days,
                 "title": "Aggregated fleet consumption", "ylabel": r"$C^{\mathrm{base}}$ [MW]", "hide_yticks": True},
            ],
            figsize=(12, 10),
        )
        plot_forecast_features(spot_vals, temp_vals, wind_vals, forecast_day)

    # --- Extract scenario sample banks ---
    print("\n" + "=" * 60)
    print("EXTRACTING SCENARIOS")
    print("=" * 60)
    EAM_sample = extract_raw_scenarios_matrix(EAM_price_DK1, price_days, value_col="aFRR_ActivatedEUR", return_format="dict")
    Imbalance_sample = extract_raw_scenarios_matrix(imbalance_DK1, price_days, value_col="ImbalancePriceEUR", return_format="dict")
    cons_sample = extract_raw_scenarios_matrix(df_charger, cons_days, value_col="power", return_format="dict")
    cons_per_box_sample = extract_raw_scenarios_matrix(df_charger_per_box, cons_days, value_col="power", box_col="chargeBoxId", return_format="dict")

    # Sample banks as plain arrays (per-box consumption scaled like before).
    EAM_bank = EAM_sample["data"]
    IMB_bank = Imbalance_sample["data"]
    CONS_agg_bank = cons_sample["data"]
    CONS_per_box_bank = cons_per_box_sample["data"] / 1000

    print("\nScenario matrix shapes:")
    print(f"  EAM: {EAM_bank.shape}, dates: {EAM_sample['dates']}")
    print(f"  Imbalance: {IMB_bank.shape}")
    print(f"  Consumption: {CONS_agg_bank.shape}")
    print(f"  Consumption per box: {CONS_per_box_bank.shape}, boxes: {len(cons_per_box_sample['boxes'])}")
    print(f"  Spot price: {spot_vals.shape}")

    # --- Extract the target day's ACTUAL realized profiles (incl. day-ahead price) ---
    actuals = extract_target_day_actuals(
        forecast_day=forecast_day,
        EAM_price_DK1=EAM_price_DK1,
        imbalance_DK1=imbalance_DK1,
        df_charger_per_box=df_charger_per_box,
        spot_price_DK1=spot_price_DK1,
        save_dir=(save_dir if save_to_disk else None),
        zone=zone,
    )

    # --- Optionally persist the sample banks ---
    if save_to_disk:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"EAM_price_samples_{zone}.npy"), EAM_bank)
        np.save(os.path.join(save_dir, f"Imbalance_price_samples_{zone}.npy"), IMB_bank)
        np.save(os.path.join(save_dir, "agg_consumption_samples.npy"), CONS_agg_bank)
        np.save(os.path.join(save_dir, "consumption_per_box_sample.npy"), CONS_per_box_bank)
        print("\nSample banks saved.")

    # --- Scenario selection statistics ---
    print("\n" + "=" * 60)
    print("SCENARIO SELECTION STATISTICS")
    print("=" * 60)
    price_stats = get_similar_day_stats(price_days, price_prof)
    cons_stats = get_similar_day_stats(cons_days, cons_prof)
    print(f"\nPrice scenarios (n={price_stats['n_scenarios']}): "
          f"range {price_stats['date_range']}, weekend share {price_stats['weekend_share']:.1%}")
    print(f"Consumption scenarios (n={cons_stats['n_scenarios']}): "
          f"range {cons_stats['date_range']}, weekend share {cons_stats['weekend_share']:.1%}")

    return {
        # Sample banks (in-memory), keyed to match scenarios.build_scenarios.
        "EAM_samples": EAM_bank,
        "imbalance_samples": IMB_bank,
        "consumption_per_box_samples": CONS_per_box_bank,
        "agg_consumption_samples": CONS_agg_bank,
        # Target-day actuals.
        "actuals": actuals,
        # Selection diagnostics.
        "price_days": price_days,
        "cons_days": cons_days,
        "spot_vals": spot_vals,
        "temp_vals": temp_vals,
        "wind_vals": wind_vals,
    }


#%%
# =========================================================
# Main execution (standalone)
# =========================================================
if __name__ == "__main__":
    run_pipeline(forecast_day="2025-12-19", zone=ZONE, k_neighbors=17)