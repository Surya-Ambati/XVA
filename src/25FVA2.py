import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Exposure Data (unchanged)
def generate_exposure_data(notional, fixed_rate, market_rate, years=12):
    times = np.linspace(0, years, years * 12 + 1)
    epe, ene = [], []
    for t in times:
        if fixed_rate > market_rate:
            epe.append(notional * (fixed_rate - market_rate) * (1 - t/years) * 1e6)
            ene.append(-notional * (fixed_rate - market_rate) * 0.1 * 1e6)
        else:
            epe.append(notional * (fixed_rate - market_rate) * 0.1 * 1e6)
            ene.append(-notional * (fixed_rate - market_rate) * (1 - t/years) * 1e6)
    return times, epe, ene

# Burgard-Kjaer Model (Standard Close-Out, No Collateral)
def burgard_kjaer_model(epe, ene, risk_free_rate, lambda_b, lambda_c, r_b, r_c, funding_spread, years=12, strategy="one-bond"):
    times = np.linspace(0, years, len(epe))
    dt = times[1] - times[0]
    discount = np.exp(-(risk_free_rate + lambda_b + lambda_c) * np.array(times))
    
    # CVA and DVA (same for all strategies)
    cva = -(1 - r_c) * sum(lambda_c * d * max(e, 0) * dt for e, d in zip(epe, discount)) / 1e6
    dva = -(1 - r_b) * sum(lambda_b * d * min(e, 0) * dt for e, d in zip(ene, discount)) / 1e6
    
    # FCA depends on strategy
    if strategy == "perfect":
        fca = 0  # Perfect replication
    elif strategy == "no-shortfall":
        fca = -(1 - r_b) * sum(lambda_b * d * max(e, 0) * dt for e, d in zip(epe, discount)) / 1e6
    else:  # One-bond
        r_f = risk_free_rate + funding_spread
        discount_f = np.exp(-(r_f + lambda_c) * np.array(times))
        fva = -sum(funding_spread * d * (e + e) * dt for e, d in zip(epe, discount_f)) / 1e6  # Symmetric FVA
    
    # Total XVA
    if strategy == "one-bond":
        xva = cva + fva
    else:
        xva = cva + dva + fca
    
    return cva, dva, fca if strategy != "one-bond" else fva, xva

# Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        # Left Panel: Inputs
        dbc.Col([
            html.H3("Inputs", className="text-center"),
            dbc.Label("Notional ($M)"),
            dcc.Input(id="notional", type="number", value=100, className="mb-3"),
            dbc.Label("Fixed Rate (In-the-Money Swap, %)"),
            dcc.Input(id="fixed-rate-in", type="number", value=5, className="mb-3"),
            dbc.Label("Market Rate (%)"),
            dcc.Input(id="market-rate", type="number", value=3, className="mb-3"),
            dbc.Label("Risk-Free Rate (%)"),
            dcc.Input(id="risk-free-rate", type="number", value=2, className="mb-3"),
            dbc.Label("Funding Spread (%)"),
            dcc.Input(id="funding-spread", type="number", value=1, className="mb-3"),
            dbc.Label("Bank Default Intensity (λ_B, %)"),
            dcc.Input(id="lambda-b", type="number", value=2, className="mb-3"),
            dbc.Label("Counterparty Default Intensity (λ_C, %)"),
            dcc.Input(id="lambda-c", type="number", value=1, className="mb-3"),
            dbc.Label("Bank Recovery Rate (R_B, %)"),
            dcc.Input(id="r-b", type="number", value=40, className="mb-3"),
            dbc.Label("Counterparty Recovery Rate (R_C, %)"),
            dcc.Input(id="r-c", type="number", value=40, className="mb-3"),
            dbc.Label("Hedging Strategy"),
            dcc.Dropdown(id="strategy", options=[
                {"label": "Perfect Replication", "value": "perfect"},
                {"label": "No Shortfall", "value": "no-shortfall"},
                {"label": "One-Bond", "value": "one-bond"}
            ], value="one-bond", className="mb-3"),
            dbc.Button("Calculate", id="calc-button", color="primary", className="mt-3"),
        ], width=4, className="bg-light p-3"),

        # Right Panel: Outputs and Graphs
        dbc.Col([
            html.H3("Results", className="text-center"),
            html.Div(id="results-output", className="mb-4"),
            dcc.Tabs([
                dcc.Tab(label="Exposure Profile", children=[dcc.Graph(id="exposure-graph")]),
                dcc.Tab(label="XVA Components", children=[dcc.Graph(id="xva-components-graph")]),
            ])
        ], width=8)
    ])
], fluid=True)

@app.callback(
    [Output("results-output", "children"),
     Output("exposure-graph", "figure"),
     Output("xva-components-graph", "figure")],
    [Input("calc-button", "n_clicks")],
    [State("notional", "value"),
     State("fixed-rate-in", "value"),
     State("market-rate", "value"),
     State("risk-free-rate", "value"),
     State("funding-spread", "value"),
     State("lambda-b", "value"),
     State("lambda-c", "value"),
     State("r-b", "value"),
     State("r-c", "value"),
     State("strategy", "value")]
)
def update_dashboard(n_clicks, notional, fixed_rate_in, market_rate, risk_free_rate, funding_spread, lambda_b, lambda_c, r_b, r_c, strategy):
    if n_clicks is None:
        return "Click Calculate to see results", go.Figure(), go.Figure()

    risk_free_rate = risk_free_rate / 100
    funding_spread = funding_spread / 100
    fixed_rate_in = fixed_rate_in / 100
    market_rate = market_rate / 100
    lambda_b = lambda_b / 100
    lambda_c = lambda_c / 100
    r_b = r_b / 100
    r_c = r_c / 100

    # Exposure Profile
    times, epe_in, ene_in = generate_exposure_data(notional, fixed_rate_in, market_rate, years=12)

    # Burgard-Kjaer Model
    cva, dva, fca_or_fva, xva = burgard_kjaer_model(epe_in, ene_in, risk_free_rate, lambda_b, lambda_c, r_b, r_c, funding_spread, years=12, strategy=strategy)

    # Results
    results = html.Div([
        html.P(f"CVA: ${cva:.2f}M"),
        html.P(f"DVA: ${dva:.2f}M"),
        html.P(f"FCA/FVA: ${fca_or_fva:.2f}M"),
        html.P(f"Total XVA: ${xva:.2f}M"),
    ])

    # Graphs
    # Exposure Profile
    exposure_fig = go.Figure(data=[
        go.Scatter(x=times, y=epe_in, mode='lines', name='EPE'),
        go.Scatter(x=times, y=ene_in, mode='lines', name='ENE')
    ])
    exposure_fig.update_layout(
        title="Exposure Profile (In-the-Money Swap)",
        xaxis_title="Time (Years)",
        yaxis_title="Exposure ($M)",
        annotations=[dict(x=0, y=-5, text="Note: EPE/ENE drive XVA terms. Interpretation: Positive exposure leads to CVA/FCA.", showarrow=False, yshift=-50)]
    )

    # XVA Components
    xva_fig = go.Figure(data=[
        go.Bar(name="CVA", x=["XVA Components"], y=[cva]),
        go.Bar(name="DVA", x=["XVA Components"], y=[dva]),
        go.Bar(name="FCA/FVA", x=["XVA Components"], y=[fca_or_fva]),
        go.Bar(name="Total XVA", x=["XVA Components"], y=[xva])
    ])
    xva_fig.update_layout(
        title=f"XVA Components ({strategy.capitalize()} Strategy)",
        yaxis_title="Adjustment ($M)",
        annotations=[dict(x=0, y=-5, text=f"Note: Strategy: {strategy}. Interpretation: FCA/FVA varies by hedging approach.", showarrow=False, yshift=-50)]
    )

    return results, exposure_fig, xva_fig

if __name__ == "__main__":
    app.run(debug=True)