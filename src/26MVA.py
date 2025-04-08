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

# CCP Funding Costs and MVA
def ccp_fva_mva(epe, ene, funding_spread, initial_margin, volatility_buffer, default_fund, years=12):
    times = np.linspace(0, years, len(epe))
    dt = times[1] - times[0]
    
    # Variation Margin (assumed rehypothecatable, no FVA)
    variation_margin = np.array(epe) + np.array(ene)  # Net exposure
    
    # Initial Margin, Volatility Buffer, Default Fund (non-rehypothecatable)
    total_margin = initial_margin + volatility_buffer + default_fund
    mva = sum(total_margin * funding_spread * dt for _ in times) / 1e6  # Funding cost of margin
    
    # FVA (if variation margin not fully covered, simplified)
    fva_cost = sum(max(e, 0) * funding_spread * dt for e in epe) / 1e6
    fva_benefit = sum(min(e, 0) * funding_spread * dt for e in ene) / 1e6
    net_fva = fva_cost + fva_benefit
    
    return net_fva, mva, variation_margin

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
            dcc.Input(id="funding-spread", type="number", value=1, className="mb-3"),
            dbc.Label("Initial Margin ($M)"),
            dcc.Input(id="initial-margin", type="number", value=5, className="mb-3"),
            dbc.Label("Volatility Buffer ($M)"),
            dcc.Input(id="volatility-buffer", type="number", value=2, className="mb-3"),
            dbc.Label("Default Fund Contribution ($M)"),
            dcc.Input(id="default-fund", type="number", value=1, className="mb-3"),
            dbc.Button("Calculate", id="calc-button", color="primary", className="mt-3"),
        ], width=4, className="bg-light p-3"),

        # Right Panel: Outputs and Graphs
        dbc.Col([
            html.H3("Results", className="text-center"),
            html.Div(id="results-output", className="mb-4"),
            dcc.Tabs([
                dcc.Tab(label="Exposure Profile", children=[dcc.Graph(id="exposure-graph")]),
                dcc.Tab(label="CCP Funding Costs", children=[dcc.Graph(id="ccp-costs-graph")]),
                dcc.Tab(label="Collateral Flows (Fig 10.1)", children=[dcc.Graph(id="collateral-flows-graph")]),
            ])
        ], width=8)
    ])
], fluid=True)

@app.callback(
    [Output("results-output", "children"),
     Output("exposure-graph", "figure"),
     Output("ccp-costs-graph", "figure"),
     Output("collateral-flows-graph", "figure")],
    [Input("calc-button", "n_clicks")],
    [State("notional", "value"),
     State("fixed-rate", "value"),
     State("market-rate", "value"),
     State("funding-spread", "value"),
     State("initial-margin", "value"),
     State("volatility-buffer", "value"),
     State("default-fund", "value")]
)
def update_dashboard(n_clicks, notional, fixed_rate, market_rate, funding_spread, initial_margin, volatility_buffer, default_fund):
    if n_clicks is None:
        return "Click Calculate to see results", go.Figure(), go.Figure(), go.Figure()

    fixed_rate = fixed_rate / 100
    market_rate = market_rate / 100
    funding_spread = funding_spread / 100

    # Exposure Profile
    times, epe, ene = generate_exposure_data(notional, fixed_rate, market_rate, years=12)

    # CCP Funding Costs and MVA
    net_fva, mva, variation_margin = ccp_fva_mva(epe, ene, funding_spread, initial_margin * 1e6, volatility_buffer * 1e6, default_fund * 1e6)

    # Results
    results = html.Div([
        html.P(f"Net FVA (Variation Margin): ${net_fva:.2f}M"),
        html.P(f"MVA (Initial Margin + Volatility Buffer + Default Fund): ${mva:.2f}M"),
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
        annotations=[dict(x=0, y=-5, text="Note: EPE/ENE drive variation margin. Interpretation: Basis for FVA.", showarrow=False, yshift=-50)]
    )

    # CCP Funding Costs
    ccp_costs_fig = go.Figure(data=[
        go.Bar(name="FVA (Variation Margin)", x=["Funding Costs"], y=[net_fva]),
        go.Bar(name="MVA (IM + VB + DF)", x=["Funding Costs"], y=[mva])
    ])
    ccp_costs_fig.update_layout(
        title="CCP Funding Costs (10.1.1)",
        yaxis_title="Cost ($M)",
        annotations=[dict(x=0, y=-1, text="Note: IM, VB, DF are non-rehypothecatable. Interpretation: MVA dominates costs.", showarrow=False, yshift=-50)]
    )

    # Collateral Flows (Figure 10.1)
    collateral_flows_fig = go.Figure()
    collateral_flows_fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, label=dict(text="Corporate"), fillcolor="lightblue")
    collateral_flows_fig.add_shape(type="rect", x0=2, y0=0, x1=3, y1=1, label=dict(text="Bank"), fillcolor="lightgray")
    collateral_flows_fig.add_shape(type="rect", x0=4, y0=0, x1=5, y1=1, label=dict(text="CCP"), fillcolor="lightgreen")
    collateral_flows_fig.add_shape(type="rect", x0=4.5, y0=1.5, x1=5.5, y1=2, label=dict(text="Margin Account"), fillcolor="lightyellow")
    collateral_flows_fig.add_shape(type="line", x0=1, y0=0.5, x1=2, y1=0.5, line=dict(color="blue", width=2), label=dict(text="Collateral"))
    collateral_flows_fig.add_shape(type="line", x0=3, y0=0.5, x1=4, y1=0.5, line=dict(color="blue", width=2), label=dict(text="Variation Margin"))
    collateral_flows_fig.add_shape(type="line", x0=3, y0=1, x1=4.5, y1=1.5, line=dict(color="red", width=2), label=dict(text="IM, VB, DF"))
    collateral_flows_fig.update_layout(
        title="Collateral Flows with CCP (Fig 10.1)",
        showlegend=False,
        annotations=[dict(x=2.5, y=-0.5, text="Note: IM, VB, DF require borrowing. Interpretation: Leads to MVA.", showarrow=False)]
    )

    return results, exposure_fig, ccp_costs_fig, collateral_flows_fig

if __name__ == "__main__":
    app.run(debug=True)