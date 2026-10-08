# Optimal Bidding Strategy for EV Aggregators in the aFRR Energy Activation Market

Code accompanying the paper **"Optimal bidding strategy for EV aggregators in the aFRR market"**
by *TODO: Author 1, Author 2* (Technical University of Denmark, *TODO: year*).

The repository implements a **stochastic receding-horizon bidding model** for an aggregator of EV
chargers participating in the Nordic aFRR **energy activation market (EAM)**, together with the
**similar-day scenario generation** and the **sensitivity analyses** reported in the paper.

📄 Paper: *TODO: DOI / arXiv link*
📊 Figures: all figures in the paper are produced by the scripts in this repository.

---

## ⚠️ Data availability

**The measurement data used in the paper is confidential and is not included in this repository.**

The EV charging data (per-charger meter values) was provided by an industrial partner under a
non-disclosure agreement, and the cleaning code depends on an internal, non-public package.
Consequently **the results in the paper cannot be reproduced bit-for-bit from this repository.**

What *is* provided:

| | |
|---|---|
| ✅ | The complete model, scenario-generation and analysis code as it was run for the paper |
| ✅ | A precise specification of every input array — file, shape, resolution and units ([`data/README.md`](data/README.md)) |
| ✅ | A synthetic-data generator (`make_synthetic_data.py`) so the full pipeline runs end-to-end out of the box |
| ❌ | The charging, aFRR, imbalance and weather data used in the paper |

The publicly available series (aFRR activation prices, imbalance prices, day-ahead prices, weather)
can be re-downloaded from the sources listed in [`data/README.md`](data/README.md); only the
charging data is irreplaceable. If you supply your own charging data in the documented format, the
pipeline runs unmodified.

---

## Repository layout

```
.
├── src/
│   ├── config.py                 # all paths and global settings — start here
│   ├── data_preparation.py       # raw → cleaned 15-min / 1-s series  (needs private raw data)
│   ├── scenario_generation.py    # similar-day (k-NN) scenario selection for a target day
│   ├── scenarios.py              # assembles in-sample / out-of-sample scenario sets
│   ├── bidding_model.py          # ⭐ the optimization model (Gurobi)
│   ├── sensitivity.py            # parameter sweeps (α, θ, |Ω|) and multi-day averaging
│   ├── plotting.py               # result figures
│   ├── plotting_style.py         # IEEE single-column figure style (new_fig / save_fig)
│   ├── visualization.py          # scenario-selection diagnostic figures
│   ├── clustering_utils.py       # k-search, cluster metrics, temporal features
│   └── utils.py                  # scenario-matrix extraction from time series
├── scripts/
│   ├── run_case_study.py         # ⭐ reproduces the main case-study figures
│   ├── run_sensitivity.py        # reproduces the sensitivity figures
│   └── make_synthetic_data.py    # generates fake data so the repo runs without private data
├── figures/
│   └── information_exchange.tex  # TikZ source for the model-structure figure
├── data/
│   └── README.md                 # ⭐ input data specification (formats, shapes, units, sources)
├── results/                      # written at runtime (git-ignored)
├── requirements.txt
├── LICENSE
└── README.md
```

⭐ = the files most readers will want.

---

## Method overview

```
                 ┌──────────────────────────┐
 raw market &    │  data_preparation.py     │   cleaned 15-min / 1-s series
 charging data ─►│  (private inputs)        ├─►  EAM, imbalance, spot, load, weather
                 └──────────────────────────┘
                                │
                                ▼
   for one target day  ┌──────────────────────────────┐
                       │  scenario_generation.py      │   k historical "analogue" days,
                       │  k-NN on day-ahead drivers   ├─► selected on spot+wind (prices)
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
                       │  MILP per step (Gurobi)      │   realize, update rebound backlog
                       └──────────────────────────────┘
                                │
                                ▼
                       plotting.py / sensitivity.py → figures in results/
```

**Time structure.** The operating day is 96 quarter-hours (MTUs). Quarters are grouped into
*flexibility windows*; all rebound energy must be recovered before its window ends. At step `q_k`
the model solves over the remaining quarters of the window but **commits only the bid for `q_k`**,
then rolls forward. Activation is settled at **1-second resolution** against the realized EAM price,
which is why the EAM arrays are 86 400 steps long while everything else is 96.

---

## Installation

```bash
git clone https://github.com/TODO/TODO.git
cd TODO
python -m venv .venv && source .venv/bin/activate      # or: conda create -n afrr python=3.11
pip install -r requirements.txt
```

**Gurobi.** The model is built with `gurobipy` and requires a licence. The free pip install includes
a size-limited licence that is **too small for this model** (the MILP has `|Q_solve| × |Ω|` binaries —
thousands at `|Ω| = 250`). Academic users can obtain a free full licence from
[gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/).
The model is a standard MILP and can be ported to another solver via Pyomo/PuLP if needed.

Tested with Python 3.11, Gurobi 11, NumPy 1.26, pandas 2.2, scikit-learn 1.5.

---

## Quick start (no private data required)

```bash
python scripts/make_synthetic_data.py       # writes plausible fake series to data/synthetic/
python scripts/run_case_study.py --synthetic --day 2025-12-18 --scenarios 50
```

This runs the complete pipeline — scenario generation, receding-horizon optimization, settlement and
plotting — on synthetic data. **The numbers are meaningless**; the purpose is to verify the
installation, show the expected data shapes, and let readers step through the model.

With real data in place, drop the flag:

```bash
python scripts/run_case_study.py --day 2025-12-18 --scenarios 250
```

---

## Reproducing the paper

All parameters are set at the top of each script (or via CLI flags). Runtimes are for
`|Ω| = 250` in-sample scenarios on a laptop with Gurobi 11.

| Paper item | Script | Command | ~Runtime |
|---|---|---|---|
| Fig. *TODO* — scenario selection | `run_case_study.py` | `--day 2025-12-18 --plot-scenarios` | < 1 min |
| Fig. *TODO* — bids, activation & load | `run_case_study.py` | `--day 2025-12-18` | ~25 min |
| Fig. *TODO* — profit decomposition | `run_case_study.py` | `--day 2025-12-18` | (same run) |
| Fig. *TODO* — per-charger response | `run_case_study.py` | `--day 2025-12-18 --chargers 3 12 25 94` | (same run) |
| Fig. *TODO* — sensitivity to α | `run_sensitivity.py` | `--param alpha --values 0.5 0.75 0.8 0.9 1.0` | ~2 h |
| Fig. *TODO* — sensitivity to θ (5-day mean) | `run_sensitivity.py` | `--param theta --multiday` | ~10 h |
| Fig. *TODO* — sensitivity to \|Ω\| | `run_sensitivity.py` | `--param n_omega --values 10 40 60 100 150 200 240 270` | ~4 h |
| Fig. *TODO* — model structure | — | `latexmk -pdf figures/information_exchange.tex` | — |
| Tab. *TODO* — hyper-parameter grid | `run_sensitivity.py` | `--grid` | ~12 h |

Figures are written to `results/figures/` as PDF in the IEEE single-column format
(3.5 in wide, matching fonts), driven by `src/plotting_style.py`.

---

## Notation: paper ↔ code

The code deliberately mirrors the paper's symbols so that constraints can be read side by side.
Constraint comments in `bidding_model.py` carry the paper's equation labels (`eq:c1`, `eq:c2`, …).

| Paper | Code | Meaning | Unit |
|---|---|---|---|
| $p^{\text{bid}}_q$ | `p_bid_q` | committed capacity bid in quarter *q* | MW |
| $\lambda^{\text{bid}}_q$ | `lambda_bid_q` | committed price bid | €/MWh |
| $\lambda^{\text{EAM}}_{q\omega}$ | `EAM_scenarios_q` | aFRR activation price (quarterly mean of 1-s series) | €/MWh |
| $\lambda^{\text{imb}}_{q\omega}$ | `min_imbalance_scenarios` | imbalance price | €/MWh |
| $\lambda^{\text{DA}}_q$ | `spot_price` | day-ahead price | €/MWh |
| $P^{\text{base}}_{q\omega}$ | `consumption_scenarios` | fleet baseline power | MW |
| $e^{\text{act}}, e^{\text{reb}}$ | `e_act`, `e_reb` | scheduled activation / rebound energy | MWh |
| $x_{q\omega}$ | `x` | activation trigger (binary, big-M) | – |
| $s_\omega$ | `s` | energy-balance slack | MWh |
| $B_c$ | `B` | rebound backlog per charger | MWh |
| $H_c, H^{\text{agg}}$ | `H_c`, `H_agg` | rebound headroom, charger / fleet | MW |
| $\alpha$ | `alpha` | share of baseline capacity offered, α ∈ (0,1] | – |
| $\theta$ | `theta` | risk-aversion percentile of the imbalance price | % |
| $\sigma$ | `sigma` | slack penalty | €/MWh |
| $\Omega$ | `Omega`, `n_omega` | in-sample scenario set | – |
| $\mathcal{C}$ | `N_charge` | chargers in the fleet | – |
| $\mathcal{Q}^{\text{win}}$ | `WINDOWS` | flexibility windows (lists of quarter indices) | – |

---

## Key parameters

Set in `src/config.py` or passed on the command line.

| Parameter | Default | Description |
|---|---|---|
| `zone` | `"DK1"` | Danish bidding zone |
| `W` (`n_omega`) | `250` | in-sample scenarios (price × load combinations) |
| `W_out` | `5` | held-out out-of-sample combinations |
| `k_neighbors` | `17` | analogue days selected per series |
| `alpha` | `0.8` | share of baseline load offered as capacity |
| `theta` | `50` | percentile of the imbalance price used in the price-bid floor |
| `fee` | `0` | activation fee added to the price floor [€/MWh] |
| `WINDOWS` | 4 windows | quarter indices per flexibility window |
| `sigma` | `1e5` | slack penalty; large enough to make the balance effectively hard |
| `M` | `1e6` | big-M for the activation trigger |
| `time_limit` | `120` | per-step Gurobi time limit [s] |
| `cons_zero_cap` | `0.15` | drop analogue days with more than this fraction of zero quarters |
| `LOOKBACK_Q` | `12` | quarters in the 3-hour power-cap look-back |

---

## Notes and known limitations

- **`sigma` is a penalty, not a hard constraint.** The window energy balance can be violated at cost
  `sigma` per MWh. Check `results["backlog_end_window"]` — a non-negligible residual backlog means
  the fleet could not physically recover the activated energy inside the window.
- **Per-step solve time.** Each of the 96 steps solves its own MILP over the remaining window with
  `|Q_solve| × |Ω|` binaries. `|Ω|` drives runtime roughly linearly; the `time_limit` of 120 s is
  reached for the first steps of long windows, and the solution is then accepted at the incumbent.
- **Normalization.** Consumption scenarios are scaled so the mean fleet load is 1 MW. Absolute
  profits therefore scale linearly with fleet size; the paper reports normalized values.
- **Activation is modelled as a price threshold**, not as a TSO merit-order clearing with volume
  constraints. Bids are assumed accepted whenever the realized EAM price meets the price bid.
- **Rebound allocation is proportional** to each charger's post-rebound load, and is capped by the
  peak baseline power over the preceding 3 hours. It is a heuristic disaggregation of an
  aggregate-level decision, not a per-EV feasibility check against actual departure times.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{TODO2026afrr,
  title   = {Optimal bidding strategy for EV aggregators in the aFRR market},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO},
  doi     = {TODO}
}
```

## License

Code released under the MIT License — see [`LICENSE`](LICENSE).
The data used in the paper is not covered by this license and is not distributed here.

## Acknowledgments

*TODO: industrial data partner, funding, supervisors.*

## Contact

Questions about the model or the data format: *TODO: name, email* — or open an issue.
