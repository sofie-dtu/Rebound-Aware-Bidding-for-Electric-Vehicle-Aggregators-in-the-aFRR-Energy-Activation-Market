"""Utility functions for scenario extraction and data processing."""

import numpy as np
import pandas as pd


def extract_raw_scenarios_matrix(df, similar_days, value_col='power', box_col=None, return_format='dataframe'):
    """
    Extract raw data for selected days, preserving date information.
    
    Automatically detects timestep per box and fills missing timestamps with zeros.
    Supports 15-min (96) or 1-sec (86400) data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime index
    similar_days : list
        List of dates to extract
    value_col : str
        Column name to extract values from
    box_col : str, optional
        Column for per-box data. If None, treats as aggregated.
    return_format : str
        'dataframe' (default): Returns DataFrame with dates as index
        'array': Returns numpy array (backward compatible)
        'dict': Returns dict with dates and arrays
        
    Returns
    -------
    For return_format='dataframe':
        pd.DataFrame: Aggregated data with dates as index
        or dict of DataFrames per box
    For return_format='array':
        np.ndarray as before
    For return_format='dict':
        {'dates': list, 'data': np.ndarray}
    """
    similar_days = pd.to_datetime(similar_days)
    df = df.copy()

    # Detect resolution
    if box_col is None:
        diffs = df.index.to_series().diff().dropna().dt.total_seconds()
    else:
        diffs = (
            df.groupby(box_col).apply(
                lambda x: x.index.to_series().diff().dropna().dt.total_seconds().median()
            )
        )
    step = np.median(diffs)
    n_steps = 96 if step > 60 else 86400
    freq = '15min' if n_steps == 96 else '1S'
    
    # Aggregate case
    if box_col is None:
        all_days = []
        dates = []
        for d in similar_days:
            full_index = pd.date_range(
                d, d + pd.Timedelta(days=1),
                freq=freq, inclusive="left"
            )

            day = (
                df.loc[
                    (df.index >= d) & (df.index < d + pd.Timedelta(days=1)),
                    value_col
                ]
                .groupby(level=0).mean()
                .reindex(full_index, fill_value=0)
            )

            all_days.append(day.values)
            dates.append(d.date())

        if return_format == 'dataframe':
            # Return DataFrame with time indices and date columns
            time_idx = pd.date_range('00:00', periods=len(all_days[0]), freq=freq)
            return pd.DataFrame(
                np.column_stack(all_days),
                index=time_idx,
                columns=dates
            )
        elif return_format == 'dict':
            return {'dates': dates, 'data': np.stack(all_days, axis=1)}
        else:  # array
            return np.stack(all_days, axis=1)

    # Per-box case
    boxes = df[box_col].unique()
    all_days = []
    dates = []
    
    for d in similar_days:
        full_index = pd.date_range(d, d + pd.Timedelta(days=1), freq=freq, inclusive='left')
        day_data = []
        for b in boxes:
            subset = df.loc[
                (df[box_col] == b) &
                (df.index >= d) & (df.index < d + pd.Timedelta(days=1)),
                value_col
            ]
            subset = subset.groupby(subset.index).mean()
            subset = subset.reindex(full_index, fill_value=0)
            day_data.append(subset.values)
        all_days.append(np.stack(day_data, axis=1))
        dates.append(d.date())
    
    result_array = np.stack(all_days, axis=1) if all_days else None
    
    if return_format == 'dataframe':
        # Return dict of DataFrames per box, with dates as index
        time_idx = pd.date_range('00:00', periods=result_array.shape[0], freq=freq)
        result_dict = {}
        for i, box in enumerate(boxes):
            result_dict[box] = pd.DataFrame(
                result_array[:, :, i],
                index=time_idx,
                columns=dates
            )
        return result_dict
    elif return_format == 'dict':
        return {'dates': dates, 'data': result_array, 'boxes': boxes}
    else:  # array
        return result_array
