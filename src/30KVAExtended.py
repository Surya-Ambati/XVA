import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Portfolio KVA with CVA Capital Allocation and Variable Cost of Capital
def portfolio_kva_extended(notional, years, counterparties, exposures, lambda_b, r_b, risk_free_rate, gamma_k_base, phi, leverage_ratio, total_capital, k_cva_total, m_ead, m_hedge_b):
    times = np.linspace(0, years, int(years * 12) + 1)
    dt = times[1] - times[0]
    discount = np.exp(-(risk_free_rate + lambda_b) * np.array(times))

    # Total exposure for leverage ratio
    total_exposure = sum(exposures)
    leverage_capital = total_exposure / leverage_ratio
    k_leverage = leverage_capital / notional * 10000

    # Market risk capital (simplified)
    k_market = 50

    # CVA capital allocation (Equation 15.43)
    k_cva_j = []
    denominator = sum((m * ead - mhb) for m, ead, mhb in zip([1]*len(exposures), exposures, m_hedge_b))
    for m, ead, mhb in zip([1]*len(exposures), exposures, m_hedge_b):
        k_cva_j.append(((m * ead - mhb) / denominator) * k_cva_total if denominator != 0 else 0)

    # Total capital
    k_total = k_leverage + k_market + k_cva_total

    # Variable cost of capital (increases with total capital)
    gamma_k = gamma_k_base * (1 + 0.02 * k_total / 100)  # Example: 2% increase per 100 bps

    # Allocate total capital using Euler allocation (exposure-based)
    k_j = []
    for exposure in exposures:
        k_j.append(k_total * (exposure / total_exposure))

    # KVA per counterparty
    kva_j = []
    for kj in k_j:
        kva = -sum((gamma_k - phi * risk_free_rate) * kj * d * dt for d in discount) / notional * 10000
        kva_j.append(kva)

    return kva_j, k_leverage, k_market, k_cva_j, k_total, gamma_k

# Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        # Left Panel: Inputs
        dbc.Col([
            html.H3("Inputs", className="text-center"),
            dbc.Label("Total Notional ($M)"),
            dcc.Input(id="notional", type="number", value=100, className="mb-3"),
            dbc.Label("Portfolio Maturity (Years)"),
            dcc.Input(id="years", type="number", value=10, className="mb-3"),
            dbc.Label("Number of Counterparties"),
            dcc.Input(id="counterparties", type="number", value=2, className="mb-3"),
            dbc.Label("Exposures (Comma-separated, $M)"),
            dcc.Input(id="exposures", type="text", value="60,40", className="mb-3"),
            dbc.Label("Bank Default Intensity (λ_B, %)"),
            dcc.Input(id="lambda-b", type="number", value=1, className="mb-3"),
            dbc.Label("Bank Recovery Rate (R_B, %)"),
            dcc.Input(id="r-b", type="number", value=40, className="mb-3"),
            dbc.Label("Risk-Free Rate (%)"),
            dcc.Input(id="risk-free-rate", type="number", value=2, className="mb-3"),
            dbc.Label("Base Cost of Capital (γ_K, %)"),
            dcc.Input(id="gamma-k-base", type="number", value=10, className="mb-3"),
            dbc.Label("Phi (Fraction of Capital for Funding)"),
            dcc.Input(id="phi", type="number", value=0, className="mb-3"),
            dbc.Label("Leverage Ratio (%)"),
            dcc.Input(id="leverage-ratio", type="number", value=7.5, className="mb-3"),
            dbc.Label("Total Capital ($M)"),
            dcc.Input(id="total-capital", type="number", value=6, className="mb-3"),
            dbc.Label("Total CVA Capital (bps of Notional)"),
            dcc.Input(id="k-cva-total", type="number", value=50, className="mb-3"),
            dbc.Label("M*EAD (Comma-separated, $M)"),
            dcc.Input(id="m-ead", type="text", value="50,30", className="mb-3"),
            dbc.Label("M_hedge*B (Comma-separated, $M)"),
            dcc.Input(id="m-hedge-b", type="text", value="0,0", className="mb-3"),
            dbc.Button("Calculate", id="calc-button", color="primary", className="mt-3"),
        ], width=4, className="bg-light p-3"),

        # Right Panel: Outputs and Graphs
        dbc.Col([
            html.H3("Results", className="text-center"),
            html.Div(id="results-output", className="mb-4"),
            dcc.Tabs([
                dcc.Tab(label="KVA per Counterparty", children=[dcc.Graph(id="kva-counterparty-graph")]),
                dcc.Tab(label="Capital Components", children=[dcc.Graph(id="capital-components-graph")]),
                dcc.Tab(label="CVA Capital Allocation", children=[dcc.Graph(id="cva-capital-graph")]),
            ])
        ], width=8)
    ])
], fluid=True)

@app.callback(
    [Output("results-output", "children"),
     Output("kva-counterparty-graph", "figure"),
     Output("capital-components-graph", "figure"),
     Output("cva-capital-graph", "figure")],
    [Input("calc-button", "n_clicks")],
    [State("notional", "value"),
     State("years", "value"),
     State("counterparties", "value"),
     State("exposures", "value"),
     State("lambda-b", "value"),
     State("r-b", "value"),
     State("risk-free-rate", "value"),
     State("gamma-k-base", "value"),
     State("phi", "value"),
     State("leverage-ratio", "value"),
     State("total-capital", "value"),
     State("k-cva-total", "value"),
     State("m-ead", "value"),
     State("m-hedge-b", "value")]
)
def update_dashboard(n_clicks, notional, years, counterparties, exposures, lambda_b, r_b, risk_free_rate, gamma_k_base, phi, leverage_ratio, total_capital, k_cva_total, m_ead, m_hedge_b):
    if n_clicks is None:
        return "Click Calculate to see results", go.Figure(), go.Figure(), go.Figure()

    # Parse exposures, m_ead, and m_hedge_b
    try:
        exposures = [float(x.strip()) for x in exposures.split(",")]
        m_ead = [float(x.strip()) for x in m_ead.split(",")]
        m_hedge_b = [float(x.strip()) for x in m_hedge_b.split(",")]
        if len(exposures) != counterparties or len(m_ead) != counterparties or len(m_hedge_b) != counterparties:
            return "Error: Number of exposures, M*EAD, and M_hedge*B must match number of counterparties", go.Figure(), go.Figure(), go.Figure()
    except:
        return "Error: Invalid input format", go.Figure(), go.Figure(), go.Figure()

    lambda_b = lambda_b / 100
    r_b = r_b / 100
    risk_free_rate = risk_free_rate / 100
    gamma_k_base = gamma_k_base / 100
    phi = max(0, min(1, phi))

    # Portfolio KVA Calculation
    kva_j, k_leverage, k_market, k_cva_j, k_total, gamma_k = portfolio_kva_extended(
        notional, years, counterparties, exposures, lambda_b, r_b, risk_free_rate, gamma_k_base, phi, leverage_ratio, total_capital, k_cva_total, m_ead, m_hedge_b
    )

    # Results
    results = html.Div([
        html.P(f"Total Capital Requirement: {k_total:.2f} bps of Notional"),
        html.P(f"Effective Cost of Capital (γ_K): {gamma_k*100:.2f}%"),
        html.P("KVA per Counterparty (bps of Notional):"),
        *[html.P(f"Counterparty {i+1}: {kva:.2f} bps") for i, kva in enumerate(kva_j)],
        html.P("CVA Capital per Counterparty (bps of Notional):"),
        *[html.P(f"Counterparty {i+1}: {k_cva:.2f} bps") for i, k_cva in enumerate(k_cva_j)]
    ])

    # Graphs
    # KVA per Counterparty
    kva_counterparty_fig = go.Figure(data=[
        go.Bar(name="KVA", x=[f"Counterparty {i+1}" for i in range(len(kva_j))], y=kva_j)
    ])
    kva_counterparty_fig.update_layout(
        title="KVA per Counterparty (Euler Allocation)",
        yaxis_title="KVA (bps of Notional)",
        annotations=[dict(x=0, y=-50, text="Note: Variable cost of capital. Interpretation: Reflects total capital impact.", showarrow=False, yshift=-50)]
    )

    # Capital Components
    capital_components_fig = go.Figure(data=[
        go.Bar(name="Leverage Ratio Capital", x=["Capital Components"], y=[k_leverage]),
        go.Bar(name="Market Risk Capital", x=["Capital Components"], y=[k_market]),
        go.Bar(name="CVA Capital", x=["Capital Components"], y=[k_cva_total]),
        go.Bar(name="Total Capital", x=["Capital Components"], y=[k_total])
    ])
    capital_components_fig.update_layout(
        title="Capital Components (Portfolio Level)",
        yaxis_title="Capital (bps of Notional)",
        annotations=[dict(x=0, y=-50, text="Note: Includes CVA and leverage ratio. Interpretation: Portfolio-level impact.", showarrow=False, yshift=-50)]
    )

    # CVA Capital Allocation
    cva_capital_fig = go.Figure(data=[
        go.Bar(name="CVA Capital", x=[f"Counterparty {i+1}" for i in range(len(k_cva_j))], y=k_cva_j)
    ])
    cva_capital_fig.update_layout(
        title="CVA Capital Allocation",
        yaxis_title="CVA Capital (bps of Notional)",
        annotations=[dict(x=0, y=-10, text="Note: Based on net exposure contribution. Interpretation: Proportional to M*EAD.", showarrow=False, yshift=-50)]
    )

    return results, kva_counterparty_fig, capital_components_fig, cva_capital_fig

if __name__ == "__main__":
    app.run(debug=True)