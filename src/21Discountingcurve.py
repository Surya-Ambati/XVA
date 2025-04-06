import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Dummy Data Generation
def generate_dummy_data():
    dates = [datetime.today() + timedelta(days=i*365) for i in range(5)]  # 0 to 4 years
    usd_rates = [0.02, 0.025, 0.03, 0.035, 0.04]  # USD LIBOR rates
    eur_rates = [0.015, 0.02, 0.025, 0.03, 0.035]  # EURIBOR rates
    ois_rates = [0.015, 0.018, 0.02, 0.022, 0.025]  # OIS rates
    basis_spread = 0.002  # 20bp cross-currency basis
    tenor_basis = {'3M': 0.001, '6M': 0.002}  # Tenor basis spreads
    fx_rate = 1.1  # USD/EUR FX rate
    return dates, usd_rates, eur_rates, ois_rates, basis_spread, tenor_basis, fx_rate

# 8.2 Single Curve Discounting
def single_curve_discount(dates, rates, interp_date):
    """Calculate discount factors using single curve and log-linear interpolation."""
    discount_factors = [1 / (1 + r * (d - dates[0]).days / 365) for d, r in zip(dates, rates)]
    t = (interp_date - dates[0]).days / 365
    ti = [0] + [(d - dates[0]).days / 365 for d in dates[1:]]
    
    for i in range(len(ti) - 1):
        if ti[i] <= t <= ti[i+1]:
            log_p0 = np.log(discount_factors[i])
            log_p1 = np.log(discount_factors[i+1])
            p = np.exp(((log_p1 * (t - ti[i])) / (ti[i+1] - ti[i])) + ((log_p0 * (ti[i+1] - t)) / (ti[i+1] - ti[i])))
            return p
    return discount_factors[-1]  # Extrapolate if beyond last date

# 8.3 Smooth Curve Interpolation (Monotone Convex Approximation)
def smooth_curve_interpolation(dates, rates, interp_date):
    """Simplified monotone convex interpolation."""
    t = (interp_date - dates[0]).days / 365
    ti = [0] + [(d - dates[0]).days / 365 for d in dates[1:]]
    discount_factors = [1 / (1 + r * ti[i]) for i, r in enumerate(rates)]
    
    for i in range(len(ti) - 1):
        if ti[i] <= t <= ti[i+1]:
            # Linear interpolation of forward rates, then discount
            f0 = -np.log(discount_factors[i]) / ti[i] if ti[i] > 0 else rates[0]
            f1 = -np.log(discount_factors[i+1]) / ti[i+1]
            f = f0 + (f1 - f0) * (t - ti[i]) / (ti[i+1] - ti[i])
            return np.exp(-f * t)
    return discount_factors[-1]

# 8.4 Cross-Currency Basis
def cross_currency_discount(dates, usd_rates, eur_rates, basis_spread, fx_rate, interp_date):
    """Calculate foreign (EUR) discount curve with cross-currency basis."""
    usd_df = [1 / (1 + r * (d - dates[0]).days / 365) for d, r in zip(dates, usd_rates)]
    eur_df = []
    for i, d in enumerate(dates):
        t = (d - dates[0]).days / 365
        if i == 0:
            eur_df.append(1.0)
        else:
            # Simplified bootstrap: Adjust EUR rate with basis
            f_eur = eur_rates[i-1] + basis_spread
            eur_df.append(usd_df[i-1] / (1 + f_eur * t / (i+1)))
    
    t = (interp_date - dates[0]).days / 365
    ti = [0] + [(d - dates[0]).days / 365 for d in dates[1:]]
    for i in range(len(ti) - 1):
        if ti[i] <= t <= ti[i+1]:
            return eur_df[i] + (eur_df[i+1] - eur_df[i]) * (t - ti[i]) / (ti[i+1] - ti[i])
    return eur_df[-1]

# 8.5 Multi-Curve Tenor Basis
def multi_curve_tenor(dates, base_rates, tenor_basis, tenor, interp_date):
    """Calculate projection curve for a specific tenor with basis."""
    base_df = [1 / (1 + r * (d - dates[0]).days / 365) for d, r in zip(dates, base_rates)]
    tenor_df = []
    b = tenor_basis[tenor]
    for i, d in enumerate(dates):
        t = (d - dates[0]).days / 365
        if i == 0:
            tenor_df.append(1.0)
        else:
            f_tenor = base_rates[i-1] + b
            tenor_df.append(base_df[i-1] / (1 + f_tenor * t / (i+1)))
    
    t = (interp_date - dates[0]).days / 365
    ti = [0] + [(d - dates[0]).days / 365 for d in dates[1:]]
    for i in range(len(ti) - 1):
        if ti[i] <= t <= ti[i+1]:
            return tenor_df[i] + (tenor_df[i+1] - tenor_df[i]) * (t - ti[i]) / (ti[i+1] - ti[i])
    return tenor_df[-1]

# 8.6 OIS/CSA Discounting
def ois_csa_discount(dates, ois_rates, interp_date):
    """Calculate discount factors using OIS rates."""
    return single_curve_discount(dates, ois_rates, interp_date)  # Reuse single curve logic with OIS

# 8.6.3 Multi-Currency Collateral Option
def multi_currency_collateral(dates, domestic_rates, foreign_rates, basis_adjustment, interp_date):
    """Calculate cheapest-to-deliver discount curve."""
    t = (interp_date - dates[0]).days / 365
    ti = [0] + [(d - dates[0]).days / 365 for d in dates[1:]]
    ctd_rates = [max(domestic_rates[i], foreign_rates[i] + basis_adjustment) for i in range(len(dates))]
    discount_factors = [np.exp(-r * ti[i]) for i, r in enumerate(ctd_rates)]
    
    for i in range(len(ti) - 1):
        if ti[i] <= t <= ti[i+1]:
            return discount_factors[i] + (discount_factors[i+1] - discount_factors[i]) * (t - ti[i]) / (ti[i+1] - ti[i])
    return discount_factors[-1]

# Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        # Left Panel: Inputs
        dbc.Col([
            html.H3("Inputs", className="text-center"),
            dbc.Label("Interpolation Date (YYYY-MM-DD)"),
            dcc.Input(id="interp-date", type="text", value="2026-04-05", className="mb-3"),
            dbc.Label("Domestic Rates (comma-separated, e.g., 0.02,0.025,...)"),
            dcc.Input(id="domestic-rates", type="text", value="0.02,0.025,0.03,0.035,0.04", className="mb-3"),
            dbc.Label("Foreign Rates (comma-separated)"),
            dcc.Input(id="foreign-rates", type="text", value="0.015,0.02,0.025,0.03,0.035", className="mb-3"),
            dbc.Label("OIS Rates (comma-separated)"),
            dcc.Input(id="ois-rates", type="text", value="0.015,0.018,0.02,0.022,0.025", className="mb-3"),
            dbc.Label("Cross-Currency Basis Spread"),
            dcc.Input(id="basis-spread", type="number", value=0.002, className="mb-3"),
            dbc.Label("FX Rate (USD/EUR)"),
            dcc.Input(id="fx-rate", type="number", value=1.1, className="mb-3"),
            dbc.Label("Tenor Basis (3M, 6M)"),
            dcc.Input(id="tenor-basis", type="text", value="0.001,0.002", className="mb-3"),
            dbc.Button("Calculate", id="calc-button", color="primary", className="mt-3"),
        ], width=4, className="bg-light p-3"),

        # Right Panel: Outputs and Graphs
        dbc.Col([
            html.H3("Results", className="text-center"),
            html.Div(id="results-output", className="mb-4"),
            dcc.Tabs([
                dcc.Tab(label="Single Curve", children=[dcc.Graph(id="single-curve-graph")]),
                dcc.Tab(label="Smooth Curve", children=[dcc.Graph(id="smooth-curve-graph")]),
                dcc.Tab(label="Cross-Currency", children=[dcc.Graph(id="cross-currency-graph")]),
                dcc.Tab(label="Multi-Curve Tenor", children=[dcc.Graph(id="multi-curve-graph")]),
                dcc.Tab(label="OIS/CSA", children=[dcc.Graph(id="ois-csa-graph")]),
                dcc.Tab(label="Multi-Currency Collateral", children=[dcc.Graph(id="multi-collateral-graph")]),
            ])
        ], width=8)
    ])
], fluid=True)

@app.callback(
    [Output("results-output", "children"),
     Output("single-curve-graph", "figure"),
     Output("smooth-curve-graph", "figure"),
     Output("cross-currency-graph", "figure"),
     Output("multi-curve-graph", "figure"),
     Output("ois-csa-graph", "figure"),
     Output("multi-collateral-graph", "figure")],
    [Input("calc-button", "n_clicks")],
    [State("interp-date", "value"),
     State("domestic-rates", "value"),
     State("foreign-rates", "value"),
     State("ois-rates", "value"),
     State("basis-spread", "value"),
     State("fx-rate", "value"),
     State("tenor-basis", "value")]
)
def update_dashboard(n_clicks, interp_date, domestic_rates, foreign_rates, ois_rates, basis_spread, fx_rate, tenor_basis):
    if n_clicks is None:
        return "Click Calculate to see results", *[go.Figure() for _ in range(6)]

    dates = [datetime.today() + timedelta(days=i*365) for i in range(5)]
    interp_date = datetime.strptime(interp_date, "%Y-%m-%d")
    domestic_rates = [float(r) for r in domestic_rates.split(",")]
    foreign_rates = [float(r) for r in foreign_rates.split(",")]
    ois_rates = [float(r) for r in ois_rates.split(",")]
    tenor_basis = {"3M": float(tenor_basis.split(",")[0]), "6M": float(tenor_basis.split(",")[1])}

    # Calculations
    single_df = single_curve_discount(dates, domestic_rates, interp_date)
    smooth_df = smooth_curve_interpolation(dates, domestic_rates, interp_date)
    cross_df = cross_currency_discount(dates, domestic_rates, foreign_rates, basis_spread, fx_rate, interp_date)
    tenor_df_3m = multi_curve_tenor(dates, domestic_rates, tenor_basis, "3M", interp_date)
    ois_df = ois_csa_discount(dates, ois_rates, interp_date)
    multi_df = multi_currency_collateral(dates, domestic_rates, foreign_rates, basis_spread, interp_date)

    results = html.Div([
        html.P(f"Single Curve DF: {single_df:.4f}"),
        html.P(f"Smooth Curve DF: {smooth_df:.4f}"),
        html.P(f"Cross-Currency DF: {cross_df:.4f}"),
        html.P(f"Tenor 3M DF: {tenor_df_3m:.4f}"),
        html.P(f"OIS/CSA DF: {ois_df:.4f}"),
        html.P(f"Multi-Currency Collateral DF: {multi_df:.4f}")
    ])

    # Graphs
    ti = [0] + [(d - dates[0]).days / 365 for d in dates[1:]]
    single_dfs = [single_curve_discount(dates, domestic_rates, d) for d in dates]
    smooth_dfs = [smooth_curve_interpolation(dates, domestic_rates, d) for d in dates]
    cross_dfs = [cross_currency_discount(dates, domestic_rates, foreign_rates, basis_spread, fx_rate, d) for d in dates]
    tenor_dfs = [multi_curve_tenor(dates, domestic_rates, tenor_basis, "3M", d) for d in dates]
    ois_dfs = [ois_csa_discount(dates, ois_rates, d) for d in dates]
    multi_dfs = [multi_currency_collateral(dates, domestic_rates, foreign_rates, basis_spread, d) for d in dates]

    single_fig = go.Figure(data=[go.Scatter(x=ti, y=single_dfs, mode='lines+markers', name='Discount Factors')])
    single_fig.update_layout(title="Single Curve Discount Factors", xaxis_title="Time (Years)", yaxis_title="Discount Factor",
                             annotations=[dict(x=0, y=0, text="Note: Simple log-linear interpolation. Interpretation: Shows basic time value of money.", showarrow=False, yshift=-50)])

    smooth_fig = go.Figure(data=[go.Scatter(x=ti, y=smooth_dfs, mode='lines+markers', name='Discount Factors')])
    smooth_fig.update_layout(title="Smooth Curve Discount Factors", xaxis_title="Time (Years)", yaxis_title="Discount Factor",
                             annotations=[dict(x=0, y=0, text="Note: Monotone convex approximation. Interpretation: Smoother forward rates.", showarrow=False, yshift=-50)])

    cross_fig = go.Figure(data=[go.Scatter(x=ti, y=cross_dfs, mode='lines+markers', name='EUR Discount Factors'),
                                go.Scatter(x=ti, y=single_dfs, mode='lines+markers', name='USD Discount Factors')])
    cross_fig.update_layout(title="Cross-Currency Discount Factors", xaxis_title="Time (Years)", yaxis_title="Discount Factor",
                            annotations=[dict(x=0, y=0, text="Note: Includes basis spread. Interpretation: Adjusts for currency risk.", showarrow=False, yshift=-50)])

    multi_fig = go.Figure(data=[go.Scatter(x=ti, y=tenor_dfs, mode='lines+markers', name='3M Tenor'),
                                go.Scatter(x=ti, y=single_dfs, mode='lines+markers', name='Base Curve')])
    multi_fig.update_layout(title="Multi-Curve Tenor Discount Factors (3M)", xaxis_title="Time (Years)", yaxis_title="Discount Factor",
                            annotations=[dict(x=0, y=0, text="Note: Tenor basis applied. Interpretation: Reflects tenor-specific risks.", showarrow=False, yshift=-50)])

    ois_fig = go.Figure(data=[go.Scatter(x=ti, y=ois_dfs, mode='lines+markers', name='OIS Discount Factors')])
    ois_fig.update_layout(title="OIS/CSA Discount Factors", xaxis_title="Time (Years)", yaxis_title="Discount Factor",
                          annotations=[dict(x=0, y=0, text="Note: Uses OIS rates. Interpretation: Closer to risk-free.", showarrow=False, yshift=-50)])

    multi_coll_fig = go.Figure(data=[go.Scatter(x=ti, y=multi_dfs, mode='lines+markers', name='Cheapest-to-Deliver'),
                                     go.Scatter(x=ti, y=single_dfs, mode='lines+markers', name='Domestic'),
                                     go.Scatter(x=ti, y=[1/(1+r*t) for r,t in zip(foreign_rates, ti)], mode='lines+markers', name='Foreign')])
    multi_coll_fig.update_layout(title="Multi-Currency Collateral Discount Factors", xaxis_title="Time (Years)", yaxis_title="Discount Factor",
                                 annotations=[dict(x=0, y=0, text="Note: Max of adjusted rates. Interpretation: Reflects collateral option.", showarrow=False, yshift=-50)])

    return results, single_fig, smooth_fig, cross_fig, multi_fig, ois_fig, multi_coll_fig

if __name__ == "__main__":
    app.run(debug=True)