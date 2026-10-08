"""
fleet_model_paper.py
====================

Stochastic receding-horizon bidding model for an EV aggregator in the aFRR
energy activation market (EAM). The notation follows the paper:

    Optimal bidding strategy for EV aggregators in the aFRR market.

Naming conventions (matching the paper):
    p_bid_q      capacity bid in quarter q                          [MW]
    lambda_bid_q price bid in quarter q                             [EUR/MWh]
    e_act, e_reb scheduled activation / rebound energy              [MWh]
    s            energy-balance slack per scenario                  [MWh]
    alpha        share of baseline capacity offered, alpha in (0,1]
    theta        risk-aversion percentile for the imbalance price
    sigma        slack penalty                                      [EUR/MWh]
    M            Big-M constant
    Omega        in-sample scenario set
    C            chargers in the fleet
    B            rebound backlog per charger                        [MWh]
    H_c, H_agg   rebound headroom, charger and fleet level          [MW]

Time structure:
    The operating day is divided into N_Q = 96 quarters (15-min MTUs).
    Quarters are grouped into flexibility windows; all rebound must complete
    before a window ends. Decisions roll forward one quarter at a time: at step
    q_k the model solves for Q_solve(q_k) but commits only the bid for q_k.
"""

import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# --- Fixed problem dimensions ---------------------------------------------
N_Q = 96            # quarters per day (4 x 24)
SEC_PER_Q = 900     # seconds per quarter (15 min)
QH = 0.25           # quarter-hour, in hours (MWh = MW * QH)
LOOKBACK_Q = 12     # quarters in the 3-hour power-cap look-back


def run_fleet_model(
    fee,
    spot_price,
    alpha,
    WINDOWS,
    EAM_scenarios,
    min_imbalance_scenarios,
    consumption_scenarios,
    N_charge,
    n_omega,
    theta,
    EAM_real=None,
    IMB_real=None,
    consumption_real=None,
    EAM_scenarios_out=None,
    min_imbalance_scenarios_out=None,
    consumption_scenarios_out=None,
    w_real=None,
    M=1e6,
    sigma=1e5,
    time_limit=120,
    compute_profit=True,
):
    """
    Run the receding-horizon bidding model over all flexibility windows.

    Bids are optimized on the in-sample scenarios and then settled against a
    realization. The realization is the actual target day when EAM_real /
    IMB_real / consumption_real are provided; otherwise it falls back to the
    out-of-sample scenario indexed by w_real.

    Parameters
    ----------
    fee : float
        Per-MWh activation fee f [EUR/MWh] added to the price-bid floor.
    spot_price : np.ndarray, shape (N_Q,)
        Day-ahead price lambda_da_q [EUR/MWh].
    alpha : float
        Share of in-sample baseline capacity offered, alpha in (0, 1].
    WINDOWS : list[list[int]]
        Flexibility windows, each a list of quarter indices.
    EAM_scenarios : np.ndarray, shape (N_Q * SEC_PER_Q, n_omega)
        In-sample EAM price scenarios at second resolution.
    min_imbalance_scenarios : np.ndarray, shape (N_Q, n_omega)
        In-sample imbalance price scenarios lambda_imb_q_omega [EUR/MWh].
    consumption_scenarios : np.ndarray, shape (N_Q, n_omega)
        In-sample fleet baseline power P_q_omega [MW].
    N_charge : sequence
        Charger indices (the fleet C).
    n_omega : int
        Number of in-sample scenarios |Omega|.
    theta : float
        Risk-aversion percentile for the imbalance-price floor.
    EAM_real : np.ndarray, shape (N_Q * SEC_PER_Q,), optional
        Actual target-day EAM price at second resolution.
    IMB_real : np.ndarray, shape (N_Q,), optional
        Actual target-day imbalance price.
    consumption_real : np.ndarray, shape (N_Q, |C|), optional
        Actual target-day per-charger baseline power [MW].
    EAM_scenarios_out, min_imbalance_scenarios_out, consumption_scenarios_out :
        np.ndarray, optional
        Out-of-sample scenarios; used only as a fallback realization together
        with w_real when the actual-day arrays are not provided.
    w_real : int, optional
        Index of the out-of-sample scenario to realize when no actual day is
        given.
    M : float, optional
        Big-M constant.
    sigma : float, optional
        Slack penalty in the objective.
    time_limit : float, optional
        Per-solve Gurobi time limit [s].
    compute_profit : bool, optional
        If True, compute the realized profit breakdown.

    Returns
    -------
    dict
        Committed bids, realized energies, per-charger allocations, the
        realized profit breakdown, and diagnostics.
    """
    # =====================================================================
    # Sets and realization data
    # =====================================================================
    Omega = range(n_omega)
    n_C = len(N_charge)

    # Choose the realization: actual target day if given, else OOS scenario.
    if EAM_real is not None and IMB_real is not None and consumption_real is not None:
        EAM_real = np.asarray(EAM_real)
        IMB_real = np.asarray(IMB_real)
        P_real = np.asarray(consumption_real)
    elif EAM_scenarios_out is not None and w_real is not None:
        EAM_real = EAM_scenarios_out[:, w_real]           # second resolution
        IMB_real = min_imbalance_scenarios_out[:, w_real]
        P_real = consumption_scenarios_out[:, w_real, :]  # per-charger baseline
    else:
        raise ValueError(
            "Provide either (EAM_real, IMB_real, consumption_real) for the actual "
            "target day, or (EAM_scenarios_out, ..., w_real) for an OOS realization.")

    # Quarterly-average in-sample EAM price lambda_eam_q_omega.
    EAM_scenarios_q = EAM_scenarios.reshape(N_Q, SEC_PER_Q, n_omega).mean(axis=1)

    # =====================================================================
    # Output accumulators (one entry per quarter / charger)
    # =====================================================================
    p_bid_q = np.zeros(N_Q)            # committed capacity bid     [MW]
    lambda_bid_q = np.zeros(N_Q)       # committed price bid        [EUR/MWh]

    E_act_q = np.zeros(N_Q)            # realized activation energy [MWh]
    E_reb_q = np.zeros(N_Q)            # realized rebound energy    [MWh]
    E_act_hat_q = np.zeros(N_Q)        # expected activation energy [MWh]
    E_reb_hat_q = np.zeros(N_Q)        # expected rebound energy    [MWh]

    e_act_charger = np.zeros((N_Q, n_C))  # realized activation per charger [MWh]
    e_reb_charger = np.zeros((N_Q, n_C))  # realized rebound per charger    [MWh]

    B = np.zeros(n_C)                  # rebound backlog per charger [MWh]

    # Second-resolution realization traces.
    X_sec = np.zeros(N_Q * SEC_PER_Q)        # activation indicator
    e_act_sec = np.zeros(N_Q * SEC_PER_Q)    # activated energy   [MWh]
    p_act_sec = np.zeros(N_Q * SEC_PER_Q)    # activated power    [MW]

    backlog_end_window = np.zeros(len(WINDOWS))  # backlog at end of each window

    # =====================================================================
    # Realization step (eqs. for X_s, E_act, headroom, backlog)
    # =====================================================================
    def realize_quarter(q, backlog):
        """
        Realize quarter q against the out-of-sample scenario and update the
        per-charger backlog. Implements the realization-step equations:
        activation indicator X_s, power cap and headroom, proportional
        rebound and activation allocation, and the backlog recursion.
        """
        s0 = q * SEC_PER_Q
        s1 = (q + 1) * SEC_PER_Q
        prices = EAM_real[s0:s1]  # second-resolution EAM price for this quarter

        # --- Baseline power this quarter (MW) ---
        P_now = P_real[q, :].copy()
        P_tot = P_now.sum()

        # --- Power cap over the 3-hour look-back (peak baseline power) ---
        q_lo = max(0, q - LOOKBACK_Q)
        P_cap_c = P_real[q_lo:q + 1].max(axis=0)            # per charger
        P_cap_agg = P_real[q_lo:q + 1].sum(axis=1).max()    # fleet level

        # --- Rebound headroom H = cap - baseline ---
        H_c = np.maximum(P_cap_c - P_now, 0.0)
        H_agg = max(P_cap_agg - P_tot, 0.0)

        # --- Rebound: each charger recovers backlog up to its headroom ---
        B_MW = backlog / QH                       # backlog as power-equivalent
        P_reb_c = np.minimum(H_c, B_MW)
        if P_reb_c.sum() > H_agg:                 # scale to fleet headroom
            P_reb_c *= H_agg / P_reb_c.sum()
        P_after_reb = P_now + P_reb_c

        # --- Activation: trigger when realized EAM price reaches the bid ---
        X_s = ((prices >= lambda_bid_q[q]) & (prices > 0)).astype(float)
        E_act = (p_bid_q[q] * X_s / 3600.0).sum()   # MWh over the quarter
        P_act = E_act / QH                          # quarter-average power

        # --- Allocate activation across chargers (proportional to load) ---
        if P_after_reb.sum() > 1e-12:
            P_act_c = P_act * P_after_reb / P_after_reb.sum()
        else:
            P_act_c = np.zeros_like(P_now)

        # --- Convert to energy and update backlog ---
        E_reb_c = QH * P_reb_c
        E_act_c = QH * P_act_c
        backlog = backlog + E_act_c - E_reb_c

        # --- Store realization outputs ---
        e_act_charger[q] = E_act_c
        e_reb_charger[q] = E_reb_c
        E_act_q[q] = E_act
        E_reb_q[q] = E_reb_c.sum()
        X_sec[s0:s1] = X_s
        e_act_sec[s0:s1] = p_bid_q[q] * X_s / 3600.0
        p_act_sec[s0:s1] = p_bid_q[q] * X_s

        return backlog

    # =====================================================================
    # Rolling horizon over all flexibility windows
    # =====================================================================
    total_start = time.time()

    for win_id, Q_win in enumerate(WINDOWS):
        print(f"\n=== Window {win_id + 1}/{len(WINDOWS)}: "
              f"quarters {Q_win[0]}..{Q_win[-1]} ===")
        # Realized window-total energy carried across solves within the window.
        E_act_real = 0.0
        E_reb_real = 0.0

        for k, q_k in enumerate(Q_win):
            # Solve set Q_solve(q_k): remaining quarters in this window.
            Q_solve = Q_win[k:]

            # theta-percentile imbalance price over the solve set.
            imb_floor = float(np.percentile(
                min_imbalance_scenarios[Q_solve, :], theta))

            # Available rebound energy cap E_reb_bar_q_omega, derived from the
            # baseline and the peak power over the preceding look-back.
            E_reb_bar = {}
            for w in Omega:
                for q in Q_win:
                    q_lo = max(0, q - LOOKBACK_Q)
                    past = consumption_scenarios[q_lo:q, w]
                    curr = consumption_scenarios[q, w]
                    peak = curr if len(past) == 0 else past.max()
                    E_reb_bar[(q, w)] = max(0.0, peak - curr)  # [MW]

            # -------------------------------------------------------------
            # Build the optimization model for step q_k
            # -------------------------------------------------------------
            m = gp.Model(f"win{win_id}_q{q_k}")
            m.Params.OutputFlag = 0
            m.Params.TimeLimit = time_limit
            m.Params.DualReductions = 0

            # Decision variables over Q_solve.
            p_bid = m.addVars(Q_solve, lb=0.0, name="p_bid")                # MW
            lambda_bid = m.addVars(Q_solve, lb=0.0, ub=1e4, name="lambda_bid")  # EUR/MWh
            x = m.addVars([(q, w) for q in Q_solve for w in Omega],
                          vtype=GRB.BINARY, name="x")                       # trigger
            e_act = m.addVars([(q, w) for q in Q_solve for w in Omega],
                              lb=0.0, name="e_act")                         # MWh
            e_reb = m.addVars([(q, w) for q in Q_solve for w in Omega],
                              lb=0.0, name="e_reb")                         # MWh
            s = m.addVars(Omega, lb=0.0, name="s")                          # MWh slack

            # Objective: expected activation revenue net of rebound cost and
            # slack penalty (eq:obj). Spreads are taken against the spot price.
            m.setObjective(
                (1.0 / n_omega) * gp.quicksum(
                    gp.quicksum(
                        e_act[q, w] * (EAM_scenarios_q[q, w] - spot_price[q])
                        for q in Q_solve)
                    - gp.quicksum(
                        e_reb[q, w] * (min_imbalance_scenarios[q, w] - spot_price[q])
                        for q in Q_solve)
                    - sigma * s[w]
                    for w in Omega
                ),
                GRB.MAXIMIZE,
            )

            for q in Q_solve:
                # (eq:c2) Price-bid floor: cover day-ahead opportunity cost plus
                # the theta-percentile expected rebound (imbalance) cost.
                m.addConstr(
                    lambda_bid[q] >= spot_price[q] + imb_floor + fee,
                    name=f"price_floor[{q}]")

                for w in Omega:
                    # (eq:c1) Capacity capped at a share alpha of baseline.
                    m.addConstr(
                        p_bid[q] <= alpha * consumption_scenarios[q, w],
                        name=f"cap[{q},{w}]")

                    # (eq:c3)-(eq:c4) Big-M trigger: x = 1 iff EAM price >= bid.
                    diff = EAM_scenarios_q[q, w] - lambda_bid[q]
                    m.addConstr(-M * (1 - x[q, w]) <= diff, name=f"trig_lo[{q},{w}]")
                    m.addConstr(M * x[q, w] >= diff, name=f"trig_hi[{q},{w}]")

                    # (eq:c5) When triggered, deliver one quarter-hour of the bid.
                    m.addConstr(
                        e_act[q, w] == p_bid[q] * QH * x[q, w],
                        name=f"act[{q},{w}]")

                    # (eq:c6) Rebound capped by available rebound energy.
                    m.addConstr(
                        e_reb[q, w] <= QH * E_reb_bar[(q, w)],
                        name=f"reb_cap[{q},{w}]")

            # Expected energies from the two committed-but-unrealized quarters
            # Q_prev2(q_k) = {q_k - 2, q_k - 1} that lie inside this window.
            E_act_prev2 = 0.0
            E_reb_prev2 = 0.0
            for q_prev in (q_k - 2, q_k - 1):
                if q_prev in Q_win:
                    E_act_prev2 += E_act_hat_q[q_prev]
                    E_reb_prev2 += E_reb_hat_q[q_prev]

            # (eq:c7 / eq:Etot / eq:Etotreb) Energy balance: the window-total
            # rebound must cover the window-total activation up to slack s. Each
            # total sums the realized energy over Q_real, the expected energy
            # over Q_prev2, and the scheduled energy over the full Q_solve, so
            # all activated energy is recovered before the window ends.
            for w in Omega:
                m.addConstr(
                    E_reb_real + E_reb_prev2
                    + gp.quicksum(e_reb[q, w] for q in Q_solve)
                    >=
                    E_act_real + E_act_prev2
                    + gp.quicksum(e_act[q, w] for q in Q_solve)
                    - s[w],
                    name=f"energy_balance[{w}]")

            # -------------------------------------------------------------
            # Solve and commit the bid for q_k
            # -------------------------------------------------------------
            m.optimize()

            if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                p_bid_q[q_k] = p_bid[q_k].X
                lambda_bid_q[q_k] = lambda_bid[q_k].X
                # (eq:Eexp / eq:Erebexp) Expected energies = scenario averages.
                E_act_hat_q[q_k] = (1.0 / n_omega) * sum(e_act[q_k, w].X for w in Omega)
                E_reb_hat_q[q_k] = (1.0 / n_omega) * sum(e_reb[q_k, w].X for w in Omega)
                print(f"  step q={q_k:2d} | committed p_bid={p_bid_q[q_k]:7.3f} MW, "
                      f"lambda_bid={lambda_bid_q[q_k]:8.2f} EUR/MWh, "
                      f"E_act_hat={E_act_hat_q[q_k]:.4f} MWh")
            else:
                print(f"  step q={q_k:2d} | solve status {m.status}")

            # Realize the quarter that is now two steps old (q_k - 2).
            q_realized = q_k - 2
            if q_realized >= Q_win[0]:
                B = realize_quarter(q_realized, B)
                E_act_real += E_act_q[q_realized]
                E_reb_real += E_reb_q[q_realized]

        # Flush the last two committed quarters at the end of the window.
        for q_realized in Q_win[-2:]:
            B = realize_quarter(q_realized, B)
        backlog_end_window[win_id] = B.sum()

    total_runtime = (time.time() - total_start) / 60.0

    print(f"\nTotal runtime: {total_runtime:.2f} minutes")
    print("Energy summary:")
    print(f"  Total activation: {E_act_q.sum():.3f} MWh")
    print(f"  Total rebound:    {E_reb_q.sum():.3f} MWh")
    print(f"  Remaining backlog: {B.sum():.3f} MWh")
    # =====================================================================
    # Out-of-sample profit breakdown (realized scenario omega_real)
    # =====================================================================
    if compute_profit:
        # --- Activation side ---
        # aFRR revenue: activated energy settled at the realized EAM price.
        aFRR_revenue_q = (e_act_sec * EAM_real).reshape(N_Q, SEC_PER_Q).sum(axis=1)
        # Day-ahead opportunity cost of the energy not consumed.
        spot_cost_q = spot_price * E_act_q
        profit_act_q = aFRR_revenue_q - spot_cost_q

        # --- Rebound side ---
        # Imbalance energy = rebound in excess of activation in the same quarter.
        E_imb_q = np.maximum(E_reb_q - E_act_q, 0.0)
        imbalance_cost_q = E_imb_q * IMB_real          # settled at imbalance price
        spot_gain_q = E_imb_q * spot_price             # paid by retailer at spot
        profit_reb_q = imbalance_cost_q - spot_gain_q  # net rebound cost

        # --- Totals ---
        aFRR_revenue_total = aFRR_revenue_q.sum()
        spot_cost_total = spot_cost_q.sum()
        imbalance_cost_total = imbalance_cost_q.sum()
        spot_gain_total = spot_gain_q.sum()

        profit_act_total = aFRR_revenue_total - spot_cost_total
        profit_reb_total = imbalance_cost_total - spot_gain_total
        profit_net_total = profit_act_total - profit_reb_total
        profit_net_q = profit_act_q - profit_reb_q

        # --- Consumer cost impact (baseline vs. actual at spot prices) ---
        baseline_consumption_q = P_real.sum(axis=1) * QH      # MWh per quarter
        baseline_cost_q = baseline_consumption_q * spot_price
        actual_consumption_q = baseline_consumption_q - E_act_q + E_reb_q
        actual_cost_q = actual_consumption_q * spot_price
        consumer_cost_diff_q = actual_cost_q - baseline_cost_q

        # --- Print breakdown ---
        print("\nProfit breakdown (out-of-sample):")
        print(f"  aFRR revenue:           {aFRR_revenue_total:+12,.2f} EUR")
        print(f"  Spot opportunity cost:  {-spot_cost_total:+12,.2f} EUR")
        print(f"  Activation net:         {profit_act_total:+12,.2f} EUR")
        print(f"  Imbalance cost:         {-imbalance_cost_total:+12,.2f} EUR")
        print(f"  Spot opportunity gain:  {spot_gain_total:+12,.2f} EUR")
        print(f"  Rebound net:            {-profit_reb_total:+12,.2f} EUR")
        print(f"  Total net profit:       {profit_net_total:+12,.2f} EUR")
    else:
        profit_act_q = profit_reb_q = profit_net_q = None
        profit_act_total = profit_reb_total = profit_net_total = None
        aFRR_revenue_q = aFRR_revenue_total = None
        spot_cost_q = spot_cost_total = None
        imbalance_cost_q = imbalance_cost_total = None
        spot_gain_q = spot_gain_total = None
        E_imb_q = None
        baseline_consumption_q = baseline_cost_q = None
        actual_consumption_q = actual_cost_q = consumer_cost_diff_q = None

    # =====================================================================
    # Results
    # =====================================================================
    return {
        # Committed bids.
        "p_bid_q": p_bid_q,
        "lambda_bid_q": lambda_bid_q,
        # Realized and expected energies.
        "E_act_q": E_act_q,
        "E_reb_q": E_reb_q,
        "E_act_hat_q": E_act_hat_q,
        "E_reb_hat_q": E_reb_hat_q,
        # Per-charger realized allocation.
        "e_act_charger": e_act_charger,
        "e_reb_charger": e_reb_charger,
        # Second-resolution traces.
        "X_sec": X_sec,
        "e_act_sec": e_act_sec,
        "p_act_sec": p_act_sec,
        # Profit breakdown.
        "profit_act_q": profit_act_q,
        "profit_reb_q": profit_reb_q,
        "profit_net_q": profit_net_q,
        "profit_act_total": profit_act_total,
        "profit_reb_total": profit_reb_total,
        "profit_net_total": profit_net_total,
        "aFRR_revenue_q": aFRR_revenue_q,
        "aFRR_revenue_total": aFRR_revenue_total,
        "spot_cost_q": spot_cost_q,
        "spot_cost_total": spot_cost_total,
        "imbalance_cost_q": imbalance_cost_q,
        "imbalance_cost_total": imbalance_cost_total,
        "spot_gain_q": spot_gain_q,
        "spot_gain_total": spot_gain_total,
        "E_imb_q": E_imb_q,
        # Consumer cost impact.
        "baseline_consumption_q": baseline_consumption_q,
        "baseline_cost_q": baseline_cost_q,
        "actual_consumption_q": actual_consumption_q,
        "actual_cost_q": actual_cost_q,
        "consumer_cost_diff_q": consumer_cost_diff_q,
        "spot_price": spot_price,
        # Diagnostics.
        "runtime_min": total_runtime,
        "backlog_end_window": backlog_end_window,
    }