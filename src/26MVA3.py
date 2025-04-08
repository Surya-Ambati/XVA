import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.linear_model import LinearRegression

# Exposure Data with Longstaff-Schwartz (simplified)
def longstaff_schwartz_exposure(notional, fixed_rate, market_rate, years=12, paths=1000):
    times = np.linspace(0, years, years * 12 + 1)
    np.random.seed(42)
    swap_rates = np.random.normal(fixed_rate, 0.01, (len(times), paths))
    values = notional * (swap_rates - market_rate) * 1e6
    epe, ene = [], []
    for t in range(len(times)):
        X = np.vstack([swap_rates[t], swap_rates[t]**2]).T
        y = values[t]
        reg = LinearRegression().fit(X, y)
        predicted_values = reg.predict(X)
        epe.append(np.mean(np.maximum(predicted_values, 0)))
        ene.append(np.mean(np.minimum(predicted_values, 0)))
    return times, epe, ene

# LSAC for VaR and Initial Margin
def lsac_var(notional, fixed_rate, market_rate, years=12, paths=1000, var_percentile=0.99, shocks=1294):
    times, epe, ene = longstaff_schwartz_exposure(notional, fixed_rate, market_rate, years, paths)
    np.random.seed(42)
    
    # Simulate VaR shocks (simplified)
    shocks_data = np.random.uniform(-0.3, 0.35, shocks)  # -30% to +35% shocks
    initial_margin = []
    
    for t in range(len(times)):
        # Base portfolio value (using Longstaff-Schwartz)
        swap_rates = np.random.normal(fixed_rate, 0.01, paths)
        X = np.vstack([swap_rates, swap_rates**2]).T
        reg = LinearRegression().fit(X, notional * (swap_rates - market_rate) * 1e6)
        base_value = np.mean(reg.predict(X))
        
        # Apply shocks to state variables (swap rates)
        shocked_values = []
        for shock in shocks_data:
            shocked_rates = swap_rates * (1 + shock)
            X_shocked = np.vstack([shocked_rates, shocked_rates**2]).T
            shocked_value = np.mean(reg.predict(X_shocked))
            shocked_values.append(shocked_value - base_value)
        
        # Calculate VaR
        shocked_values = np.sort(shocked_values)
        var_index = int(var_percentile * len(shocked_values))
        var = -shocked_values[var_index]  # Loss at 99th percentile
        initial_margin.append(var)
    
    return times, initial_margin

# MVA with No Shortfall Strategy (from previous)
def mva_no_shortfall(initial_margin, funding_spread, lambda_b, r_b, s_i, years=12):
    times = np.linspace(0, years, int(years * 12) + 1)
    dt = times[1] - times[0]
    discount = np.exp(-(0.02 + lambda_b) * np.array(times))
    s_f = (1 - r_b) * lambda_b
    mva = -sum((s_f - s_i) * im * d * dt for im, d in zip(initial_margin, discount)) / 1e6
    return mva

# Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        # Left Panel: Inputs
        dbc.Col([
            html.H3("Inputs", className="text-center"),
            dbc.Label("Notional ($M)"),
            dcc.Input(id="notional", type="number", value=100, className="mb-3"),
            dbc.Label("Fixed Rate (Swap, %)"),
            dcc.Input(id="fixed-rate", type="number", value=5, className="mb-3"),
            dbc.Label("Market Rate (%)"),
            dcc.Input(id="market-rate", type="number", value=3, className="mb-3"),
            dbc.Label("Funding Spread (%)"),
            dcc.Input(id="funding-spread", type="number", value=1.2, className="mb-3"),
            dbc.Label("Bank Default Intensity (λ_B, %)"),
            dcc.Input(id="lambda-b", type="number", value=2, className="mb-3"),
            dbc.Label("Bank Recovery Rate (R_B, %)"),
            dcc.Input(id="r-b", type="number", value=40, className="mb-3"),
            dbc.Label("Initial Margin Spread (s_I, %)"),
            dcc.Input(id="s-i", type="number", value=0.5, className="mb-3"),
            dbc.Label("Portfolio Type"),
            dcc.Dropdown(id="portfolio-type", options=[
                {"label": "Off-Market (90% Payer)", "value": "off_90"},
                {"label": "Balanced (50% Payer)", "value": "balanced"},
                {"label": "Off-Market (10% Payer)", "value": "off_10"}
            ], value="off_90", className="mb-3"),
            dbc.Button("Calculate", id="calc-button", color="primary", className="mt-3"),
        ], width=4, className="bg-light p-3"),

        # Right Panel: Outputs and Graphs
        dbc.Col([
            html.H3("Results", className="text-center"),
            html.Div(id="results-output", className="mb-4"),
            dcc.Tabs([
                dcc.Tab(label="Initial Margin Profile (LSAC)", children=[dcc.Graph(id="im-graph")]),
                dcc.Tab(label="MVA Costs (Case Study)", children=[dcc.Graph(id="mva-costs-graph")]),
            ])
        ], width=8)
    ])
], fluid=True)

@app.callback(
    [Output("results-output", "children"),
     Output("im-graph", "figure"),
     Output("mva-costs-graph", "figure")],
    [Input("calc-button", "n_clicks")],
    [State("notional", "value"),
     State("fixed-rate", "value"),
     State("market-rate", "value"),
     State("funding-spread", "value"),
     State("lambda-b", "value"),
     State("r-b", "value"),
     State("s-i", "value"),
     State("portfolio-type", "value")]
)
def update_dashboard(n_clicks, notional, fixed_rate, market_rate, funding_spread, lambda_b, r_b, s_i, portfolio_type):
    if n_clicks is None:
        return "Click Calculate to see results", go.Figure(), go.Figure()

    fixed_rate = fixed_rate / 100
    market_rate = market_rate / 100
    funding_spread = funding_spread / 100
    lambda_b = lambda_b / 100
    r_b = r_b / 100
    s_i = s_i / 100

    # Exposure and Initial Margin with LSAC
    times, initial_margin = lsac_var(notional, fixed_rate, market_rate, years=12)

    # MVA with No Shortfall Strategy
    mva = mva_no_shortfall(initial_margin, funding_spread, lambda_b, r_b, s_i, years=12)

    # Case Study Results (Table 10.2)
    if portfolio_type == "off_90":
        exposure = 26.4
        fva_bps = 115
        mva_bps = 53
    elif portfolio_type == "balanced":
        exposure = 2.6
        fva_bps = 0
        mva_bps = 2
    else:  # off_10
        exposure = -8.9
        fva_bps = -113
        mva_bps = 56

    # Results
    results = html.Div([
        html.P(f"Expected MVA (LSAC): ${mva:.2f}M"),
        html.P(f"Portfolio Type: {portfolio_type}"),
        html.P(f"Exposure: {exposure}% of Notional"),
        html.P(f"FVA: {fva_bps} bps of Notional"),
        html.P(f"MVA: {mva_bps} bps of Notional"),
    ])

    # Graphs
    # Initial Margin Profile
    im_fig = go.Figure(data=[
        go.Scatter(x=times, y=np.array(initial_margin) / 1e6, mode='lines', name='Initial Margin')
    ])
    im_fig.update_layout(
        title="Initial Margin Profile (LSAC, 10.3.3)",
        xaxis_title="Time (Years)",
        yaxis_title="Initial Margin ($M)",
        annotations=[dict(x=0, y=-0.5, text="Note: LSAC with VaR shocks. Interpretation: Efficient IM calculation.", showarrow=False, yshift=-50)]
    )

    # MVA Costs (Case Study)
    mva_costs_fig = go.Figure(data=[
        go.Bar(name="FVA (bps)", x=["FVA/MVA"], y=[fva_bps]),
        go.Bar(name="MVA (bps)", x=["FVA/MVA"], y=[mva_bps])
    ])
    mva_costs_fig.update_layout(
        title=f"MVA Costs (Case Study, {portfolio_type}, 10.3.4)",
        yaxis_title="Cost (bps of Notional)",
        annotations=[dict(x=0, y=-10, text="Note: Basel III VaR specs. Interpretation: MVA significant even for balanced portfolios.", showarrow=False, yshift=-50)]
    )

    return results, im_fig, mva_costs_fig

if __name__ == "__main__":
    app.run(debug=True)