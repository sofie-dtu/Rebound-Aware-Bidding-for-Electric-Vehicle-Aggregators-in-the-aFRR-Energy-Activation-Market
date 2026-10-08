# Rebound-Aware Bidding for Electric Vehicle Aggregators in the aFRR Energy Activation Market

Code accompanying the paper **"Rebound-Aware Bidding for Electric Vehicle Aggregators in the aFRR Energy Activation Market"**
by Sofie Davidsen, Torine R. Herstad, and Jalal Kazempour
(Technical University of Denmark, 2026).

The repository implements a **stochastic receding-horizon bidding model** for an aggregator of EV
chargers participating in the Nordic aFRR **energy activation market (EAM)**, together with the
**similar-day scenario generation**, a **synthetic charging-data generator**, and the sensitivity
analyses reported in the paper.

📄 Paper: *TODO: DOI / arXiv link*

---

## Data and reproducibility

**Only the EV charging data is confidential. The market and weather data is public and is included
in this repository.**

| Input | | Resolution | In this repo |
|---|---|---|---|
| aFRR activation prices (EAM) | 🌍 public | 1 s | `data/EAM.zip` |
| Imbalance prices | 🌍 public | 15 min | `data/Imbalance/` |
| Day-ahead (spot) prices | 🌍 public | 1 h | `data/SpotPrice/` |
| Weather (temperature, wind) | 🌍 public | 1 h | `data/Weather/` |
| Per-charger EV consumption | 🔒 **private** | 15 min | generated — see below |

The EV charging data was provided by an industrial partner under a non-disclosure agreement and
cannot be published. In its place the repository includes `src/make_synthetic_charging.py`, which
generates a synthetic fleet with the structure and resolution the model requires, aligned to the
date range of the market data shipped here.

Because the market side is the real market and only the fleet is substituted, a run here reproduces
the **modelling workflow and the qualitative behaviour**, not the paper's numbers:

| Carries over | Does not carry over |
|---|---|
| When bids are placed and at what price | Absolute profit in € |
| How activation and rebound interact across a flexibility window | The exact analogue days selected |
| The direction and rough shape of the α, θ and \|Ω\| sensitivities | Figures matching the paper line for line |
| Whether the rebound backlog clears inside each window | |

If you have your own charging data in the format given in
[`data/README.md`](data/README.md), the pipeline runs on it unmodified.

---

## Getting started

Generate the charging data and run the model:

```bash
python src/make_synthetic_charging.py     # writes data/Consumption/*.csv
python main.py                            # runs the case study
```

`make_synthetic_charging.py` needs no arguments. It reads the date range out of the market data in
`data/`, takes the daily temperature from `data/Weather/Weather_clean.csv` so the analogue-day
search has real signal to match on, and writes the two consumption files.

**The EAM file is zipped.** `data/EAM.zip` unzip the file.

**Gurobi.** The model uses `gurobipy` and needs a licence.
[gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/).

---

## Repository layout

```
.
├── main.py                       # entry point — runs the case study and the sweeps
│
├── src/
│   ├── bidding_model.py          # stochastic receding-horizon optimization (Gurobi MILP)
│   ├── scenario_generation.py    # similar-day (k-NN) analogue selection + data loading
│   ├── scenarios.py              # assembles in-sample / out-of-sample scenario sets
│   ├── make_synthetic_charging.py# generates the synthetic EV charging data
│   ├── sensitivity.py            # parameter sweeps (α, θ, |Ω|) and multi-day averaging
│   ├── clustering_utils.py       # analogue-day statistics and temporal features
│   ├── utils.py                  # scenario-matrix extraction from the time series
│   ├── funcitons1.py             # result figures
│   ├── IEEE_plotting_functions.py# IEEE single-column figure style (new_fig / save_fig)
│   ├── visualization.py          # scenario-selection diagnostic figures
│   └── Figures.py                # the flexibility-window / decision-timing figure
│
├── data/
│   ├── EAM.zip                   # aFRR activation prices,          (public)
│   ├── Imbalance/                # imbalance prices,                (public)
│   ├── SpotPrice/                # day-ahead prices,                (public)
│   ├── Weather/                  # temperature and wind,            (public)
│   └── Consumption/              # EV charging data — generated
│
├── data_README.md                 # data specification: formats, shapes, units, sources
└── README.md
```

---

## Method overview

```
   data/ (public, shipped here)                  src/make_synthetic_charging.py
   EAM · imbalance · spot · weather              (or your own charging data)
                    │                                        │
                    └────────────────┬───────────────────────┘
                                     ▼
   for one target day  ┌──────────────────────────────┐
                       │  scenario_generation.py      │   k historical "analogue" days,
                       │  k-NN on day-ahead drivers   ├─► picked on spot + wind (prices)
                       │  + calendar features         │   and temperature (load)
                       └──────────────────────────────┘
                                     │
                                     ▼
                       ┌──────────────────────────────┐
                       │  scenarios.py                │   Ω  in-sample  (price × load pairs)
                       │  IS/OOS split, filtering     ├─►  Ω' out-of-sample
                       └──────────────────────────────┘   + the target day's realization
                                     │
                                     ▼
                       ┌──────────────────────────────┐
                       │  bidding_model.py            │   for every quarter q:
                       │  receding horizon, 96 MTUs   ├─► commit (p_bid_q, λ_bid_q),
                       │  one MILP per step (Gurobi)  │   realize, update rebound backlog
                       └──────────────────────────────┘
                                     │
                                     ▼
                   funcitons1.py / sensitivity.py → figures and results
```

**Time structure.** The operating day is 96 quarter-hours (MTUs), grouped into *flexibility
windows*; all rebound energy must be recovered before the end of its window. At decision step `q_k`
the model considers the remaining quarters of the window but **commits only the bid for `q_k`**,
then observes the realization, updates the rebound backlog, and rolls forward.

Activation is settled at **1-second resolution** against the realized EAM price. This is why the EAM
arrays carry 86 400 values per day while every other series carries 96.

---

## Consumption-data generation

`src/make_synthetic_charging.py` builds the fleet one charger at a time, as sessions rather than as
a smooth aggregate profile, because the model depends on that structure:

```
   charger population, arrival/dwell distributions, temperature
        │
        ▼
   individual charging sessions   (arrival time, dwell, rated power, taper)
        │
        ▼
   per-charger 15-min profiles    Consumption_perBox_15min.csv   [kW]
        │
        ▼
   fleet aggregation              Consumption_agg_15min.csv      [kW]
        │
        ▼
   scenario_generation.py → normalised baseline consumption scenarios
```

It is a *model* of a fleet, not a sample from one: it carries no information about the real fleet,
and numbers produced from it should not be reported as results.

---

## Notation: paper ↔ code

The code mirrors the paper's symbols so the constraints can be read alongside the formulation.
Constraint comments in `bidding_model.py` carry the paper's equation labels (`eq:c1`, `eq:c2`, …).

| Paper | Code | Meaning | Unit |
|---|---|---|---|
| $p^{\mathrm{bid}}_q$ | `p_bid_q` | committed capacity bid in quarter *q* | MW |
| $\lambda^{\mathrm{bid}}_q$ | `lambda_bid_q` | committed price bid | €/MWh |
| $\lambda^{\mathrm{EAM}}_{q\omega}$ | `EAM_scenarios_q` | aFRR activation price (quarterly mean of the 1-s series) | €/MWh |
| $\lambda^{\mathrm{imb}}_{q\omega}$ | `min_imbalance_scenarios` | imbalance price | €/MWh |
| $\lambda^{\mathrm{DA}}_q$ | `spot_price` | day-ahead price | €/MWh |
| $P^{\mathrm{base}}_{q\omega}$ | `consumption_scenarios` | fleet baseline consumption | MW |
| $e^{\mathrm{act}}, e^{\mathrm{reb}}$ | `e_act`, `e_reb` | activation / rebound energy | MWh |
| $x_{q\omega}$ | `x` | activation trigger (binary, big-M) | – |
| $s_\omega$ | `s` | energy-balance slack | MWh |
| $B_c$ | `B` | rebound backlog per charger | MWh |
| $H_c, H^{\mathrm{agg}}$ | `H_c`, `H_agg` | rebound headroom, charger / fleet | MW |
| $\alpha$ | `alpha` | share of baseline capacity offered, α ∈ (0,1] | – |
| $\theta$ | `theta` | imbalance-price percentile | % |
| $\sigma$ | `sigma` | slack penalty | €/MWh |
| $\Omega$ | `Omega`, `n_omega` | in-sample scenario set | – |
| $\mathcal{C}$ | `N_charge` | chargers in the fleet | – |
| $\mathcal{Q}^{\mathrm{win}}$ | `WINDOWS` | flexibility windows (lists of quarter indices) | – |

---

## Key parameters

Set at the top of `main.py`.

| Parameter | Default | Description |
|---|---|---|
| `zone` | `"DK1"` | Danish bidding zone |
| `W` (`n_omega`) | `250` | in-sample scenarios (price × load combinations) |
| `W_out` | `5` | held-out out-of-sample combinations |
| `k_neighbors` | `17` | analogue days selected per series |
| `alpha` | `0.8` | share of baseline consumption offered as capacity |
| `theta` | `50` | percentile of the imbalance price used in the price-bid floor |
| `fee` | `0` | activation fee added to the price floor [€/MWh] |
| `WINDOWS` | 4 windows | quarter indices per flexibility window |
| `sigma` | `1e5` | slack penalty |
| `M` | `1e6` | big-M for the activation trigger |
| `time_limit` | `120` | per-step Gurobi time limit [s] |
| `cons_zero_cap` | `0.15` | maximum fraction of zero quarters allowed for an analogue day |
| `LOOKBACK_Q` | `12` | quarters in the 3-hour power-cap look-back |

---

## Contact

Questions about the model, the consumption-data generation, or the data format:

Sofie Davidsen — sodav@dtu.dk
