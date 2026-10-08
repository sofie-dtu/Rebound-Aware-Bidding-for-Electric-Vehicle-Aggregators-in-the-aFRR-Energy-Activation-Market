# Input data specification

Five of the six inputs are **public** and you can download them yourself. Only the per-charger EV
consumption is confidential. This document specifies exactly what the code expects, so the pipeline
can be re-run on data you assemble.

| File | Public? | Where to get it |
|---|---|---|
| `EAM/EAM_clean_{zone}.csv` | 🌍 yes | [Energi Data Service](https://www.energidataservice.dk/) — aFRR activation prices |
| `Imbalance/Imbalance_clean_{zone}.csv` | 🌍 yes | [Energi Data Service](https://www.energidataservice.dk/) — imbalance prices |
| `SpotPrice/SpotPrice_clean_{zone}.csv` | 🌍 yes |  [Energi Data Service](https://www.energidataservice.dk/) - day-ahead prices|
| `Weather/Weather_clean.csv` | 🌍 yes | [Open-Meteo](https://open-meteo.com/) |
| `Consumption/Consumption_perBox_15min.csv` | 🔒 **no** | `scripts/make_synthetic_charging.py` |
| `Consumption/Consumption_agg_15min.csv` | 🔒 **no** | `scripts/make_synthetic_charging.py` (derived from the above) |

Once the four public files are in place, generating the missing fleet is one command with no
arguments:

```bash
python scripts/make_synthetic_charging.py
```

It infers the date range from the public files and writes the two consumption files beside them.

There are two entry points to the pipeline, and you only need one of them:

---

## 1. Cleaned time series (CSV)

Expected under the directory given by `DATA_DIR` in `src/config.py`.

| File | Column | Resolution | Unit | Public source |
|---|---|---|---|---|
| `EAM/EAM_clean_{zone}.csv` | `aFRR_ActivatedEUR` | 1 s | €/MWh | Energinet / [ENDK API](https://www.energidataservice.dk/) |
| `Imbalance/Imbalance_clean_{zone}.csv` | `ImbalancePriceEUR` | 15 min | €/MWh | Energinet |
| `SpotPrice/SpotPrice_clean_{zone}.csv` | `DayAheadPriceEUR` | 1 h | €/MWh | Nord Pool / ENTSO-E |
| `Consumption/Consumption_agg_15min.csv` | `power` | 15 min | kW | **private** |
| `Consumption/Consumption_perBox_15min.csv` | `chargeBoxId`, `power` | 15 min | kW | **private** |
| `Weather/Weather_clean.csv` | `temperature_2m`, `wind_speed_10m` | 1 h | °C, m/s | [Open-Meteo](https://open-meteo.com/) |

**Common format.** First column `timestamp`, parseable by `pd.to_datetime`, used as the index.
Sorted ascending, duplicates removed, no missing rows — the cleaning step in
`data_preparation.py` resamples, interpolates and forward/back-fills to guarantee this.

**`EAM_clean`.** Only *upward* activation prices are kept; the price is set to 0 in seconds where
`aFRR_Activated <= 0`. This matters: the model triggers on `price >= lambda_bid AND price > 0`, so
non-activated seconds must be zero, not NaN and not the last observed price.

**`Consumption_perBox`.** Long format — one row per `(timestamp, chargeBoxId)`. `chargeBoxId` is a
string identifier (in our data, `"{chargeBoxId}_{connectorId}"`). Quarters where a charger reports
nothing are filled with `0`, not dropped. Power is in **kW** and is divided by 1000 in
`scenario_generation.py` to give MW.

**Weather.** Used only to select analogue days (temperature drives load, wind drives prices);
resampled to 15 min inside the pipeline.

**Coverage.** You need enough history for the k-NN analogue search to have candidates — in the paper,
`2025-10-01` to `2026-04-29`, i.e. ~7 months, for `k_neighbors = 17`. Note that candidates are
hard-filtered to the same day type (weekday vs weekend), so ~7 months of history leaves roughly 150
weekday candidates, not 210.

**Generating the fleet instead.** `scripts/make_synthetic_charging.py` writes the two consumption
files in exactly this format. It takes its date range from whichever public file it finds (preferring
`Imbalance`, then `SpotPrice`, then `Weather`) and its daily temperature from `Weather_clean.csv`,
so the generated fleet shares the real calendar and responds to real weather. Keep `--chargers` at
100 or more.

---

## 2. Scenario arrays (NumPy), if you bypass scenario generation

`run_clustering_pipeline(...)` returns, and `save_to_disk=True` writes, the following:

| Key / file | Shape | Unit | Description |
|---|---|---|---|
| `EAM_samples` | `(86400, k)` | €/MWh | 1-s EAM price, one column per analogue day |
| `imbalance_samples` | `(96, k)` | €/MWh | 15-min imbalance price per analogue day |
| `consumption_per_box_samples` | `(96, k, n_box)` | MW | per-charger baseline load per analogue day |
| `agg_consumption_samples` | `(96, k)` | MW | aggregated load (diagnostics only) |

Target-day realization (`actuals`):

| Key | Shape | Unit |
|---|---|---|
| `SPOT_actual` | `(96,)` | €/MWh — hourly day-ahead price repeated to quarters |
| `EAM_actual` | `(86400,)` | €/MWh |
| `IMB_actual` | `(96,)` | €/MWh |
| `CONS_actual` | `(96, n_box)` | MW |

`scenarios.build_scenarios(...)` then forms the Cartesian product of the *k* price days and the
*k* load days, holds out `W_out` random `(price, load)` pairs as out-of-sample, and returns the rest
as the in-sample set Ω. With `k = 17` this gives 289 combinations, of which 250 are used.

**Consumption filtering.** Analogue days whose *aggregated* profile is zero for more than
`cons_zero_cap` (default 15 %) of quarters are dropped. A day with long zero stretches would
otherwise cap every chance-constrained bid at its worst quarter, since the capacity bid is bounded by
`alpha * P_base[q, ω]` for **every** scenario ω.

**Normalization.** Consumption is rescaled so the mean fleet load equals 1 MW, separately for the
in-sample and out-of-sample sets. Profits scale linearly with fleet size.

---
