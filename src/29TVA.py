import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# XVA Calculation with Double Semi-Replication
def xva_double_semi_replication(notional, years, lambda_c, lambda_c_physical, psi, gamma_e, lambda_b, r_b, risk_free_rate, gamma_k, phi, k_u, k_r, v_minus_gc=1e6):
    times = np.linspace(0, years, int(years * 12) + 1)
    dt = times[1] - times[0]
    xi = (lambda_c - lambda_c_physical) / lambda_c  # Market price of risk
    lambda_c_tilde = psi * lambda_c + (1 - psi) * (1 - xi) * lambda_c
    discount = np.exp(-(risk_free_rate + lambda_b + lambda_c_tilde) * np.array(times))

    # Simplified profiles (bps of notional)
    v_plus = 0.01 * notional  # Simplified EPE
    v_minus = -0.005 * notional  # Simplified ENE
    r_c = 0.4  # Counterparty recovery rate

    # CVA, DVA, FCA (bps of notional)
    cva = -(1 - r_c) * sum(lambda_c_tilde * v_plus * d * dt for d in discount) / notional * 10000
    dva = -(1 - r_b) * sum(lambda_b * v_minus * d * dt for d in discount) / notional * 10000
    fca = -(1 - r_b) * sum(lambda_b * v_plus * d * dt for d in discount) / notional * 10000

    # KVA (bps of notional)
    k = k_u - psi * k_r  # Capital requirement
    kva = -sum((gamma_k - phi * risk_free_rate) * k * d * dt for d in discount) / notional * 10000

    # TVA (bps of notional)
    e = gamma_k * (k_u - k_r)  # Taxable cash flow (simplified)
    delta_e_bar = -gamma_e * v_minus_gc
    tva = -sum((gamma_e * e + lambda_c * (1 - xi) * (1 - psi) * delta_e_bar) * d * dt for d in discount) / notional * 10000

    return cva, dva, fca, kva, tva

# Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        # Left Panel: Inputs
        dbc.Col([
            html.H3("Inputs", className="text-center"),
            dbc.Label("Notional ($M)"),
            dcc.Input(id="notional", type="number", value=100, className="mb-3"),
            dbc.Label("Swap Maturity (Years)"),
            dcc.Input(id="years", type="number", value=10, className="mb-3"),
            dbc.Label("Risk-Neutral Hazard Rate (λ_C, %)"),
            dcc.Input(id="lambda-c", type="number", value=2, className="mb-3"),
            dbc.Label("Physical Hazard Rate (λ_C^P, %)"),
            dcc.Input(id="lambda-c-physical", type="number", value=1.5, className="mb-3"),
            dbc.Label("Hedge Fraction (ψ, 0 to 1)"),
            dcc.Input(id="psi", type="number", value=0.5, className="mb-3"),
            dbc.Label("Tax Rate (γ_E, %)"),
            dcc.Input(id="gamma-e", type="number", value=30, className="mb-3"),
            dbc.Label("Bank Default Intensity (λ_B, %)"),
            dcc.Input(id="lambda-b", type="number", value=1, className="mb-3"),
            dbc.Label("Bank Recovery Rate (R_B, %)"),
            dcc.Input(id="r-b", type="number", value=40, className="mb-3"),
            dbc.Label("Risk-Free Rate (%)"),
            dcc.Input(id="risk-free-rate", type="number", value=2, className="mb-3"),
            dbc.Label("Cost of Capital (γ_K, %)"),
            dcc.Input(id="gamma-k", type="number", value=10, className="mb-3"),
            dbc.Label("Phi (Fraction of Capital for Funding)"),
            dcc.Input(id="phi", type="number", value=0, className="mb-3"),
            dbc.Label("Capital Requirement (K_U, bps of Notional)"),
            dcc.Input(id="k-u", type="number", value=80, className="mb-3"),
            dbc.Label("Capital Relief (K_R, bps of Notional)"),
            dcc.Input(id="k-r", type="number", value=60, className="mb-3"),
            dbc.Button("Calculate", id="calc-button", color="primary", className="mt-3"),
        ], width=4, className="bg-light p-3"),

        # Right Panel: Outputs and Graphs
        dbc.Col([
            html.H3("Results", className="text-center"),
            html.Div(id="results-output", className="mb-4"),
            dcc.Tabs([
                dcc.Tab(label="XVA Components", children=[dcc.Graph(id="xva-components-graph")]),
                dcc.Tab(label="TVA vs. Hedge Fraction", children=[dcc.Graph(id="tva-psi-graph")]),
            ])
        ], width=8)
    ])
], fluid=True)

@app.callback(
    [Output("results-output", "children"),
     Output("xva-components-graph", "figure"),
     Output("tva-psi-graph", "figure")],  # Corrected ID to match layout
    [Input("calc-button", "n_clicks")],
    [State("notional", "value"),
     State("years", "value"),
     State("lambda-c", "value"),
     State("lambda-c-physical", "value"),
     State("psi", "value"),
     State("gamma-e", "value"),
     State("lambda-b", "value"),
     State("r-b", "value"),
     State("risk-free-rate", "value"),
     State("gamma-k", "value"),
     State("phi", "value"),
     State("k-u", "value"),
     State("k-r", "value")]
)
def update_dashboard(n_clicks, notional, years, lambda_c, lambda_c_physical, psi, gamma_e, lambda_b, r_b, risk_free_rate, gamma_k, phi, k_u, k_r):
    if n_clicks is None:
        return "Click Calculate to see results", go.Figure(), go.Figure()

    lambda_c = lambda_c / 100
    lambda_c_physical = lambda_c_physical / 100
    psi = max(0, min(1, psi))  # Ensure psi is between 0 and 1
    gamma_e = gamma_e / 100
    lambda_b = lambda_b / 100
    r_b = r_b / 100
    risk_free_rate = risk_free_rate / 100
    gamma_k = gamma_k / 100
    phi = max(0, min(1, phi))  # Ensure phi is between 0 and 1

    # XVA Calculation
    cva, dva, fca, kva, tva = xva_double_semi_replication(
        notional * 1e6, years, lambda_c, lambda_c_physical, psi, gamma_e, lambda_b, r_b, risk_free_rate, gamma_k, phi, k_u, k_r
    )

    # TVA vs. Psi
    psi_values = np.linspace(0, 1, 11)
    tva_values = []
    for p in psi_values:
        _, _, _, _, tva_p = xva_double_semi_replication(
            notional * 1e6, years, lambda_c, lambda_c_physical, p, gamma_e, lambda_b, r_b, risk_free_rate, gamma_k, phi, k_u, k_r
        )
        tva_values.append(tva_p)

    # Results
    results = html.Div([
        html.P(f"CVA: {cva:.2f} bps"),
        html.P(f"DVA: {dva:.2f} bps"),
        html.P(f"FCA: {fca:.2f} bps"),
        html.P(f"KVA: {kva:.2f} bps"),
        html.P(f"TVA: {tva:.2f} bps"),
    ])

    # Graphs
    # XVA Components
    xva_components_fig = go.Figure(data=[
        go.Bar(name="CVA", x=["XVA Components"], y=[cva]),
        go.Bar(name="DVA", x=["XVA Components"], y=[dva]),
        go.Bar(name="FCA", x=["XVA Components"], y=[fca]),
        go.Bar(name="KVA", x=["XVA Components"], y=[kva]),
        go.Bar(name="TVA", x=["XVA Components"], y=[tva])
    ])
    xva_components_fig.update_layout(
        title=f"XVA Components (ψ={psi})",
        yaxis_title="XVA (bps of Notional)",
        annotations=[dict(x=0, y=-50, text="Note: Double semi-replication. Interpretation: TVA reflects tax on unhedged risk.", showarrow=False, yshift=-50)]
    )

    # TVA vs. Psi
    tva_psi_fig = go.Figure(data=[
        go.Scatter(x=psi_values, y=tva_values, mode='lines+markers', name='TVA')
    ])
    tva_psi_fig.update_layout(
        title="TVA vs. Hedge Fraction (ψ)",
        xaxis_title="Hedge Fraction (ψ)",
        yaxis_title="TVA (bps of Notional)",
        annotations=[dict(x=0, y=min(tva_values)-5, text="Note: Varies with ψ. Interpretation: Less hedging increases TVA.", showarrow=False, yshift=-50)]
    )

    return results, xva_components_fig, tva_psi_fig

if __name__ == "__main__":
    app.run(debug=True)