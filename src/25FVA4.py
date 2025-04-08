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

# FVA with WWR/RWR
def fva_with_wwr(epe, ene, funding_spread, correlation, years=12):
    times = np.linspace(0, years, len(epe))
    dt = times[1] - times[0]
    np.random.seed(42)
    # Simulate stochastic funding spread with correlation to exposure
    exposure = np.array(epe) + np.array(ene)
    spread_shocks = np.random.normal(0, 0.005, len(times))  # 50bp volatility
    correlated_shocks = correlation * (exposure / np.max(np.abs(exposure))) + np.sqrt(1 - correlation**2) * spread_shocks
    stochastic_spread = funding_spread + correlated_shocks
    fva_cost = sum(max(e, 0) * s * dt for e, s in zip(epe, stochastic_spread)) / 1e6
    fva_benefit = sum(min(e, 0) * s * dt for e, s in zip(ene, stochastic_spread)) / 1e6
    net_fva = fva_cost + fva_benefit
    return fva_cost, fva_benefit, net_fva, stochastic_spread

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
            dbc.Label("Bank Funding Spread (%)"),
            dcc.Input(id="bank-spread", type="number", value=1, className="mb-3"),
            dbc.Label("Corporate Funding Spread (%)"),
            dcc.Input(id="corp-spread", type="number", value=2, className="mb-3"),
            dbc.Label("Correlation (WWR/RWR, -1 to 1)"),
            dcc.Input(id="correlation", type="number", value=0.5, className="mb-3"),
            dbc.Button("Calculate", id="calc-button", color="primary", className="mt-3"),
        ], width=4, className="bg-light p-3"),

        # Right Panel: Outputs and Graphs
        dbc.Col([
            html.H3("Results", className="text-center"),
            html.Div(id="results-output", className="mb-4"),
            dcc.Tabs([
                dcc.Tab(label="Exposure Profile", children=[dcc.Graph(id="exposure-graph")]),
                dcc.Tab(label="Asymmetry (Bank vs. Corporate)", children=[dcc.Graph(id="asymmetry-graph")]),
                dcc.Tab(label="WWR/RWR Impact", children=[dcc.Graph(id="wwr-graph")]),
            ])
        ], width=8)
    ])
], fluid=True)

@app.callback(
    [Output("results-output", "children"),
     Output("exposure-graph", "figure"),
     Output("asymmetry-graph", "figure"),
     Output("wwr-graph", "figure")],
    [Input("calc-button", "n_clicks")],
    [State("notional", "value"),
     State("fixed-rate", "value"),
     State("market-rate", "value"),
     State("bank-spread", "value"),
     State("corp-spread", "value"),
     State("correlation", "value")]
)
def update_dashboard(n_clicks, notional, fixed_rate, market_rate, bank_spread, corp_spread, correlation):
    if n_clicks is None:
        return "Click Calculate to see results", go.Figure(), go.Figure(), go.Figure()

    fixed_rate = fixed_rate / 100
    market_rate = market_rate / 100
    bank_spread = bank_spread / 100
    corp_spread = corp_spread / 100

    # Exposure Profile
    times, epe, ene = generate_exposure_data(notional, fixed_rate, market_rate, years=12)

    # FVA for Bank and Corporate (Asymmetry)
    bank_fva_cost, bank_fva_benefit, bank_net_fva, _ = fva_with_wwr(epe, ene, bank_spread, correlation)
    corp_fva_cost, corp_fva_benefit, corp_net_fva, stochastic_spread = fva_with_wwr(epe, ene, corp_spread, correlation)

    # Results
    results = html.Div([
        html.P(f"Bank FVA - Cost: ${bank_fva_cost:.2f}M, Benefit: ${bank_fva_benefit:.2f}M, Net: ${bank_net_fva:.2f}M"),
        html.P(f"Corporate FVA - Cost: ${corp_fva_cost:.2f}M, Benefit: ${corp_fva_benefit:.2f}M, Net: ${corp_net_fva:.2f}M"),
    ])

    # Graphs
    # Exposure Profile
    exposure_fig = go.Figure(data=[
        go.Scatter(x=times, y=epe, mode='lines', name='EPE'),
        go.Scatter(x=times, y=ene, mode='lines', name='ENE')
    ])
    exposure_fig.update_layout(
        title="Exposure Profile (Swap)",
        xaxis_title="Time (Years)",
        yaxis_title="Exposure ($M)",
        annotations=[dict(x=0, y=-5, text="Note: EPE/ENE drive FVA. Interpretation: Basis for asymmetric valuations.", showarrow=False, yshift=-50)]
    )

    # Asymmetry (Bank vs. Corporate)
    asymmetry_fig = go.Figure(data=[
        go.Bar(name="Bank FVA", x=["FVA Components"], y=[bank_fva_cost, bank_fva_benefit, bank_net_fva], text=["Cost", "Benefit", "Net"]),
        go.Bar(name="Corporate FVA", x=["FVA Components"], y=[corp_fva_cost, corp_fva_benefit, corp_net_fva], text=["Cost", "Benefit", "Net"])
    ])
    asymmetry_fig.update_layout(
        title="Asymmetry: Bank vs. Corporate FVA (9.6.1)",
        yaxis_title="FVA ($M)",
        annotations=[dict(x=0, y=-5, text="Note: Different funding spreads. Interpretation: Asymmetric valuations due to funding differences.", showarrow=False, yshift=-50)]
    )

    # WWR/RWR Impact
    wwr_fig = go.Figure(data=[
        go.Scatter(x=times, y=epe, mode='lines', name='EPE'),
        go.Scatter(x=times, y=[s * 100 for s in stochastic_spread], mode='lines', name='Stochastic Spread (%)', yaxis="y2")
    ])
    wwr_fig.update_layout(
        title="WWR/RWR Impact on FVA (9.8)",
        xaxis_title="Time (Years)",
        yaxis_title="Exposure ($M)",
        yaxis2=dict(title="Funding Spread (%)", overlaying="y", side="right"),
        annotations=[dict(x=0, y=-5, text=f"Note: Correlation = {correlation}. Interpretation: WWR increases FVA cost.", showarrow=False, yshift=-50)]
    )

    return results, exposure_fig, asymmetry_fig, wwr_fig

if __name__ == "__main__":
    app.run(debug=True)