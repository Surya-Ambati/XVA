import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import newton
from scipy.stats import norm
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import time
import random
from threading import Thread
import pandas as pd

# Simulated real-time data store for liquid counterparties
real_time_data = {
    "AAPL": {
        "maturities": [0.5, 1, 2, 3, 5, 7, 10],
        "spreads": [46, 52, 71, 94, 137, 158, 173],  # Initial spreads in bp (liquid, AA-rated)
        "interest_rate": 0.01,
        "recovery_rate": 0.4,
        "timestamp": "2025-03-31 14:30:00",
        "sector": "Technology",
        "rating": "AA",
        "region": "North America"
    }
}

# Illiquid counterparty data (no direct CDS spreads)
illiquid_data = {
    "XYZ": {
        "maturities": [0.5, 1, 2, 3, 5, 7, 10],
        "interest_rate": 0.01,
        "recovery_rate": 0.4,
        "timestamp": "2025-03-31 14:30:00",
        "sector": "Technology",
        "rating": "BBB",
        "region": "North America"
    }
}

# Simulate real-time updates to CDS spreads for liquid counterparties
def simulate_real_time_updates():
    while True:
        time.sleep(10)  # Update every 10 seconds
        for entity in real_time_data:
            real_time_data[entity]["spreads"] = [
                max(10, s + random.uniform(-5, 5)) for s in real_time_data[entity]["spreads"]
            ]
            real_time_data[entity]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

Thread(target=simulate_real_time_updates, daemon=True).start()

# --- Cross-Sectional Mapping for Illiquid Counterparties ---

def cross_sectional_mapping(illiquid_entity, liquid_data):
    liquid_entity = liquid_data["AAPL"]
    base_spreads = liquid_entity["spreads"]
    rating_map = {"AAA": 0, "AA": 1, "A": 2, "BBB": 3, "BB": 4, "B": 5, "CCC": 6}
    rating_diff = rating_map[illiquid_entity["rating"]] - rating_map[liquid_entity["rating"]]
    spread_premium = rating_diff * 50  # 50 bp per notch
    proxy_spreads = [s + spread_premium for s in base_spreads]
    return proxy_spreads

# --- Hazard Rate Models ---

def piecewise_constant_hazard(times, lambdas, t_max=None):
    t_max = times[-1] if t_max is None else t_max
    t = np.linspace(0, t_max, 1000)
    lambda_t = np.zeros_like(t)
    for i in range(len(times) - 1):
        mask = (t >= times[i]) & (t < times[i+1])
        lambda_idx = min(i, len(lambdas) - 1)
        lambda_t[mask] = lambdas[lambda_idx] if lambda_idx >= 0 else 0
    lambda_idx = len(lambdas) - 1 if lambdas else -1
    lambda_t[t >= times[-1]] = lambdas[lambda_idx] if lambda_idx >= 0 else 0
    return t, lambda_t

# --- Survival Probability and CDS Valuation ---

def survival_probability(times, lambdas, model='piecewise_constant', t_max=None):
    t_max = times[-1] if t_max is None else t_max
    t = np.linspace(0, t_max, 1000)
    if model == 'piecewise_constant':
        _, lambda_t = piecewise_constant_hazard(times, lambdas, t_max)
    else:
        raise ValueError("Only piecewise_constant model is implemented")
    
    cum_hazard = np.zeros_like(t)
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        cum_hazard[i] = cum_hazard[i-1] + lambda_t[i-1] * dt
    
    P = np.exp(-cum_hazard)
    return t, P

def initial_survival_probability(t_max, lambda_guess=0.01):
    t = np.linspace(0, t_max, 1000)
    P = np.exp(-lambda_guess * t)
    return t, P

def cds_value(notional, spread, times, r, lambdas, model='piecewise_constant'):
    alpha = 0.25
    recovery = 0.4
    premium_leg = 0
    t_fine, P_fine = survival_probability(times, lambdas, model, t_max=times[-1])
    
    for t in times[1:]:
        B = np.exp(-r * t)
        P = np.interp(t, t_fine, P_fine)
        C = spread * notional * alpha
        premium_leg += C * B * P
    
    protection_leg = 0
    for i in range(len(times) - 1):
        t_start, t_end = times[i], times[i+1]
        lambda_i = lambdas[min(i, len(lambdas)-1)] if lambdas else 0
        if lambda_i < 1e-6:
            lambda_i = 1e-6
        P_start = np.interp(t_start, t_fine, P_fine)
        P_end = np.interp(t_end, t_fine, P_fine)
        B_start = np.exp(-r * t_start)
        B_end = np.exp(-r * t_end)
        term = (lambda_i / (lambda_i + r)) * (B_start * P_start - B_end * P_end)
        protection_leg += (1 - recovery) * notional * term
    
    cds_val = premium_leg - protection_leg
    return premium_leg, protection_leg, cds_val

def bootstrap_survival_probabilities(notional, spreads, maturities, r):
    times = [0] + maturities
    lambdas = []
    steps = []
    
    for i in range(len(maturities)):
        t_start = times[i]
        t_end = times[i+1]
        spread = spreads[i]
        
        payment_times = [t for t in np.arange(t_start + 0.25, t_end + 0.25, 0.25) if t <= t_end]
        if not payment_times:
            payment_times = [t_end]
        segment_times = [0] + payment_times
        
        def cds_value_for_lambda(lambda_i):
            current_lambdas = lambdas + [lambda_i]
            _, _, cds_val = cds_value(notional, spread, segment_times, r, current_lambdas)
            return cds_val
        
        try:
            lambda_i = newton(cds_value_for_lambda, 0.01, tol=1e-6, maxiter=100)
            if lambda_i < 0:
                lambda_i = 0.001
        except RuntimeError:
            lambda_i = 0.01
        lambdas.append(lambda_i)
        
        t_fine, P_fine = survival_probability(times[:i+2], lambdas, 'piecewise_constant')
        P_end = np.interp(t_end, t_fine, P_fine)
        
        step = f"Segment {i+1} ({t_start}–{t_end} years): Spread = {spread*10000:.0f} bp, " \
               f"Lambda = {lambda_i:.4f}, Survival Probability at {t_end} years = {P_end:.4f}"
        steps.append(step)
    
    return times, lambdas, steps

# --- Analytic CVA for Interest Rate Swap ---

def black_swaption_price(notional, S, K, T, sigma, r, swap_maturities, alpha, option_type="payer"):
    annuity = sum(alpha * np.exp(-r * t) for t in swap_maturities if t > T)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "payer":
        price = notional * annuity * (S * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = notional * annuity * (K * norm.cdf(-d2) - S * norm.cdf(-d1))
    
    price *= np.exp(-r * T)
    return max(price, 0)

def analytic_cva_swap(notional, K, r, survival_times, survival_probs, swap_maturities, sigma):
    recovery = 0.4
    alpha = 0.5
    S = 0.02  # Par swap rate (simplified)
    
    cva = 0
    for i in range(len(swap_maturities) - 1):
        t_i = swap_maturities[i]
        t_i_plus_1 = swap_maturities[i + 1]
        P_ti = np.interp(t_i, survival_times, survival_probs)
        P_ti_plus_1 = np.interp(t_i_plus_1, survival_times, survival_probs)
        default_prob = P_ti - P_ti_plus_1
        remaining_maturities = [t for t in swap_maturities if t > t_i]
        vps = black_swaption_price(notional, S, K, t_i, sigma, r, swap_maturities, alpha, option_type="payer")
        cva += default_prob * vps
    
    cva *= (1 - recovery)
    return cva

# --- Analytic CVA for Interest Rate Caplet ---

def black_caplet_price(notional, f, K, T, sigma, r, alpha):
    """
    Price an interest rate caplet using Black's model.
    
    Parameters:
    - notional: Notional amount.
    - f: Forward rate f(t_s, T).
    - K: Strike rate.
    - T: Payment date (time to expiry for the option is t_s, but we discount to T).
    - sigma: Implied volatility.
    - r: Risk-free rate.
    - alpha: Day-count fraction (e.g., 0.5 for semi-annual).
    
    Returns:
    - caplet_price: Price of the caplet at t_0.
    """
    t_s = T - alpha  # Fixing date
    d1 = (np.log(f / K) + 0.5 * sigma**2 * t_s) / (sigma * np.sqrt(t_s))
    d2 = d1 - sigma * np.sqrt(t_s)
    price = notional * alpha * (f * norm.cdf(d1) - K * norm.cdf(d2))
    price *= np.exp(-r * T)  # Discount to t_0
    return max(price, 0)

def analytic_cva_caplet_unilateral(notional, K, T, f, r, sigma, alpha, survival_times, survival_probs):
    """
    Calculate the analytic unilateral CVA for an interest rate caplet.
    
    Parameters:
    - notional: Notional amount.
    - K: Strike rate.
    - T: Payment date.
    - f: Forward rate f(t_s, T).
    - r: Risk-free rate.
    - sigma: Implied volatility.
    - alpha: Day-count fraction.
    - survival_times: Times at which survival probabilities are given.
    - survival_probs: Survival probabilities at those times.
    
    Returns:
    - cva: Analytic unilateral CVA value.
    """
    recovery = 0.4
    # Survival probabilities at t_0 and T
    P_t0 = 1.0  # Assuming t_0 = 0
    P_T = np.interp(T, survival_times, survival_probs)
    default_prob = P_t0 - P_T
    # Caplet price
    caplet_price = black_caplet_price(notional, f, K, T, sigma, r, alpha)
    cva = (1 - recovery) * default_prob * caplet_price
    return cva

def analytic_cva_caplet_bilateral(notional, K, T, f, r, sigma, alpha, survival_times_c, survival_probs_c, survival_times_b, survival_probs_b):
    """
    Calculate the analytic bilateral CVA for an interest rate caplet.
    
    Parameters:
    - notional: Notional amount.
    - K: Strike rate.
    - T: Payment date.
    - f: Forward rate f(t_s, T).
    - r: Risk-free rate.
    - sigma: Implied volatility.
    - alpha: Day-count fraction.
    - survival_times_c, survival_probs_c: Counterparty survival data.
    - survival_times_b, survival_probs_b: Bank survival data.
    
    Returns:
    - bcva: Analytic bilateral CVA value.
    """
    recovery_c = 0.4
    # Caplet price
    caplet_price = black_caplet_price(notional, f, K, T, sigma, r, alpha)
    # Discretize time up to T
    times = np.arange(0, T + 0.25, 0.25)
    bcva = 0
    for i in range(len(times) - 1):
        t_i = times[i]
        t_i_plus_1 = times[i + 1]
        P_ti_c = np.interp(t_i, survival_times_c, survival_probs_c)
        P_ti_plus_1_c = np.interp(t_i_plus_1, survival_times_c, survival_probs_c)
        P_ti_b = np.interp(t_i, survival_times_b, survival_probs_b)
        default_prob_c = P_ti_plus_1_c - P_ti_c
        bcva += default_prob_c * P_ti_b
    bcva *= (1 - recovery_c) * caplet_price
    return bcva

# --- Dash App ---

app = Dash(__name__)

app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

app.layout = html.Div([
    html.Div([
        html.H1("Real-Time CVA Dashboard", style={'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id="theme-toggle",
                options=[
                    {'label': 'Light', 'value': 'plotly_white'},
                    {'label': 'Dark', 'value': 'plotly_dark'}
                ],
                value='plotly_white',
                clearable=False,
                style={'width': '100px', 'display': 'inline-block'}
            )
        ], style={'position': 'absolute', 'top': '10px', 'right': '10px'})
    ]),
    html.Div([
        html.Label("Entity ID:"),
        dcc.Dropdown(id="entity-id", options=[
            {'label': 'AAPL (Liquid)', 'value': 'AAPL'},
            {'label': 'XYZ (Illiquid)', 'value': 'XYZ'}
        ], value='AAPL'),
        html.Label("Notional Amount (USD):"),
        dcc.Input(id="notional", type="number", value=1000000, step=100000),
        html.Label("Fixed Rate for Swap (%):"),
        dcc.Input(id="fixed-rate", type="number", value=2, step=0.1),
        html.Label("Strike Rate for Caplet (%):"),
        dcc.Input(id="strike-rate", type="number", value=2, step=0.1),
        html.Label("Caplet Payment Date (Years):"),
        dcc.Input(id="caplet-T", type="number", value=1.5, step=0.5),
        html.Label("Implied Volatility (%):"),
        dcc.Input(id="volatility", type="number", value=20, step=1),
        html.Label("Initial Hazard Rate Guess (%):"),
        dcc.Input(id="lambda-guess", type="number", value=2, step=0.1),
    ]),
    dcc.Graph(id="hazard-graph"),
    dcc.Graph(id="survival-graph"),
    html.Div(id="results"),
    dcc.Interval(id="interval-component", interval=10*1000, n_intervals=0)  # Update every 10 seconds
], id="main-container")

@app.callback(
    [Output("hazard-graph", "figure"), Output("survival-graph", "figure"), Output("results", "children"),
     Output("main-container", "style")],
    [Input("entity-id", "value"), Input("notional", "value"), Input("fixed-rate", "value"),
     Input("strike-rate", "value"), Input("caplet-T", "value"), Input("volatility", "value"),
     Input("lambda-guess", "value"), Input("interval-component", "n_intervals"), Input("theme-toggle", "value")]
)
def update_graphs(entity_id, notional, fixed_rate, strike_rate, caplet_T, volatility, lambda_guess, n_intervals, theme):
    # Determine if the entity is liquid or illiquid
    if entity_id in real_time_data:
        data = real_time_data[entity_id]
        spreads = [s / 10000 for s in data["spreads"]]
        maturities = data["maturities"]
        r = data["interest_rate"]
        timestamp = data["timestamp"]
        entity_type = "Liquid"
    else:
        data = illiquid_data[entity_id]
        proxy_spreads = cross_sectional_mapping(data, real_time_data)
        spreads = [s / 10000 for s in proxy_spreads]
        maturities = data["maturities"]
        r = data["interest_rate"]
        timestamp = data["timestamp"]
        entity_type = "Illiquid (Mapped to AAPL with rating adjustment)"

    # Handle inputs
    notional = notional if notional is not None else 1000000
    K_swap = (fixed_rate if fixed_rate is not None else 2) / 100
    K_caplet = (strike_rate if strike_rate is not None else 2) / 100
    T = caplet_T if caplet_T is not None else 1.5
    sigma = (volatility if volatility is not None else 20) / 100
    lambda_guess = (lambda_guess if lambda_guess is not None else 2) / 100
    alpha = 0.5  # Semi-annual period
    f = 0.025  # Forward rate (simplified)

    # Swap payment dates (semi-annual over 10 years)
    swap_maturities = np.arange(0.5, 10.5, 0.5).tolist()

    # Initial survival probabilities (counterparty)
    t_initial, P_initial = initial_survival_probability(max(maturities), lambda_guess)
    
    # Bootstrap hazard rates and survival probabilities (counterparty)
    times, lambdas, steps = bootstrap_survival_probabilities(notional, spreads, maturities, r)
    t_bootstrapped, P_bootstrapped = survival_probability(times, lambdas, 'piecewise_constant')

    # Bank's survival probabilities (flat hazard rate of 1%)
    lambda_bank = 0.01
    t_bank, P_bank = initial_survival_probability(max(maturities), lambda_bank)

    # Hazard rate plot
    t, lambda_t = piecewise_constant_hazard(times, lambdas)
    hazard_fig = go.Figure()
    hazard_fig.add_trace(go.Scatter(x=t, y=lambda_t, mode='lines', name='Hazard Rate'))
    hazard_fig.update_layout(
        title=f"Hazard Rate Over Time (Last Updated: {timestamp})",
        xaxis_title="Time (Years)",
        yaxis_title="Hazard Rate",
        template=theme
    )

    # Survival probability plot
    survival_fig = go.Figure()
    survival_fig.add_trace(go.Scatter(x=t_initial, y=P_initial, mode='lines', name='Initial Survival Probability'))
    survival_fig.add_trace(go.Scatter(x=t_bootstrapped, y=P_bootstrapped, mode='lines', name='Bootstrapped Survival Probability'))
    survival_fig.update_layout(
        title=f"Survival Probability Over Time (Last Updated: {timestamp})",
        xaxis_title="Time (Years)",
        yaxis_title="P(tau > t)",
        template=theme
    )

    # Analytic CVA calculations
    cva_swap = analytic_cva_swap(notional, K_swap, r, t_bootstrapped, P_bootstrapped, swap_maturities, sigma)
    cva_caplet_unilateral = analytic_cva_caplet_unilateral(notional, K_caplet, T, f, r, sigma, alpha, t_bootstrapped, P_bootstrapped)
    cva_caplet_bilateral = analytic_cva_caplet_bilateral(notional, K_caplet, T, f, r, sigma, alpha, t_bootstrapped, P_bootstrapped, t_bank, P_bank)

    # Results and Interpretations
    results = [
        html.H3(f"Entity: {entity_id} ({entity_type})"),
        html.H3("Bootstrapping Steps"),
        html.P("The bootstrapping process solves for hazard rates (λ) for each segment using CDS spreads, setting V_CDS = 0. Steps:"),
        html.Ul([html.Li(step) for step in steps]),
        html.H3("Initial Survival Probabilities"),
        html.P(f"At 1 year: {np.interp(1, t_initial, P_initial):.4f}"),
        html.P(f"At 5 years: {np.interp(5, t_initial, P_initial):.4f}"),
        html.P(f"At 10 years: {np.interp(10, t_initial, P_initial):.4f}"),
        html.P(f"Interpretation: These are rough estimates based on a constant hazard rate guess (λ = {lambda_guess * 100:.2f}%). They assume a uniform default risk and do not reflect market data."),
        html.H3("Bootstrapped Survival Probabilities"),
        html.P(f"At 1 year: {np.interp(1, t_bootstrapped, P_bootstrapped):.4f}"),
        html.P(f"At 5 years: {np.interp(5, t_bootstrapped, P_bootstrapped):.4f}"),
        html.P(f"At 10 years: {np.interp(10, t_bootstrapped, P_bootstrapped):.4f}"),
        html.P("Interpretation: These are market-implied survival probabilities derived from CDS spreads. They reflect the actual default risk priced into the market, making them suitable for pricing CDS contracts and calculating CVA."),
        html.H3("Analytic CVA for Interest Rate Swap"),
        html.P(f"CVA (Unilateral, Bank Pays Fixed): ${cva_swap:.2f}"),
        html.P("Interpretation: This is the analytic CVA for a standalone interest rate swap, calculated using the swaption-based formula. It assumes no netting and uses Black's model for swaption pricing."),
        html.H3("Analytic CVA for Interest Rate Caplet"),
        html.P(f"CVA (Unilateral): ${cva_caplet_unilateral:.2f}"),
        html.P(f"CVA (Bilateral): ${cva_caplet_bilateral:.2f}"),
        html.P(f"Interpretation: The unilateral CVA is the expected loss due to counterparty default, weighted by the caplet value. The bilateral CVA adjusts for the bank's survival probability, but the DVA term is zero since the caplet value is always non-negative.")
    ]

    # Theme styling
    container_style = {
        'backgroundColor': '#ffffff' if theme == 'plotly_white' else '#1a1a1a',
        'color': '#000000' if theme == 'plotly_white' else '#ffffff',
        'padding': '20px'
    }

    return hazard_fig, survival_fig, results, container_style

if __name__ == "__main__":
    app.run(debug=True)