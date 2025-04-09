import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# KVA Calculation
def kva_calc(notional, years, counterparty_rating, phi, gamma_k, issuer_spread, r_b, risk_free_rate, hedge_type="unhedged"):
    times = np.linspace(0, years, int(years * 12) + 1)
    dt = times[1] - times[0]
    lambda_b = issuer_spread / (1 - r_b)
    lambda_c_dict = {"AAA": 0.0030, "A": 0.0075, "BB": 0.0250, "CCC": 0.0750}  # Spreads from Table 13.2
    lambda_c = lambda_c_dict[counterparty_rating]
    discount = np.exp(-(risk_free_rate + lambda_b + lambda_c) * np.array(times))

    # Simplified capital profiles (bps of notional, from Tables 13.3-13.5)
    if hedge_type == "unhedged":
        mr_dict = {"AAA": -262, "A": -256, "BB": -234, "CCC": -185}
        ccr_dict = {"AAA": -3, "A": -8, "BB": -14, "CCC": -16}
        cva_dict = {"AAA": -9, "A": -10, "BB": -22, "CCC": -87}
    elif hedge_type == "back_to_back":
        mr_dict = {"AAA": 0, "A": 0, "BB": 0, "CCC": 0}
        ccr_dict = {"AAA": -3, "A": -8, "BB": -14, "CCC": -16}
        cva_dict = {"AAA": -9, "A": -10, "BB": -22, "CCC": -87}
    else:  # ir01_hedged
        mr_dict = {"AAA": -17, "A": -20, "BB": -28, "CCC": -45}
        ccr_dict = {"AAA": -4, "A": -11, "BB": -20, "CCC": -22}
        cva_dict = {"AAA": -12, "A": -13, "BB": -31, "CCC": -127}

    # Capital profile (bps of notional)
    k_mr = mr_dict[counterparty_rating]
    k_ccr = ccr_dict[counterparty_rating]
    k_cva = cva_dict[counterparty_rating]
    k_total = k_mr + k_ccr + k_cva

    # KVA (bps of notional)
    kva = -sum((gamma_k - phi * risk_free_rate) * k_total * d * dt for d in discount)
    return kva, k_mr, k_ccr, k_cva

# Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        # Left Panel: Inputs
        dbc.Col([
            html.H3("Inputs", className="text-center"),
            dbc.Label("Notional (£M)"),
            dcc.Input(id="notional", type="number", value=100, className="mb-3"),
            dbc.Label("Swap Maturity (Years)"),
            dcc.Input(id="years", type="number", value=10, className="mb-3"),
            dbc.Label("Counterparty Rating"),
            dcc.Dropdown(id="counterparty-rating", options=[
                {"label": "AAA", "value": "AAA"},
                {"label": "A", "value": "A"},
                {"label": "BB", "value": "BB"},
                {"label": "CCC", "value": "CCC"}
            ], value="AAA", className="mb-3"),
            dbc.Label("Phi (Fraction of Capital for Funding)"),
            dcc.Input(id="phi", type="number", value=0, className="mb-3"),
            dbc.Label("Cost of Capital (γ_K, %)"),
            dcc.Input(id="gamma-k", type="number", value=10, className="mb-3"),
            dbc.Label("Issuer Spread (bps)"),
            dcc.Input(id="issuer-spread", type="number", value=100, className="mb-3"),
            dbc.Label("Bank Recovery Rate (R_B, %)"),
            dcc.Input(id="r-b", type="number", value=40, className="mb-3"),
            dbc.Label("Risk-Free Rate (%)"),
            dcc.Input(id="risk-free-rate", type="number", value=2, className="mb-3"),
            dbc.Label("Hedge Type"),
            dcc.Dropdown(id="hedge-type", options=[
                {"label": "Unhedged", "value": "unhedged"},
                {"label": "Back-to-Back Hedge", "value": "back_to_back"},
                {"label": "IR01-Hedged", "value": "ir01_hedged"}
            ], value="unhedged", className="mb-3"),
            dbc.Button("Calculate", id="calc-button", color="primary", className="mt-3"),
        ], width=4, className="bg-light p-3"),

        # Right Panel: Outputs and Graphs
        dbc.Col([
            html.H3("Results", className="text-center"),
            html.Div(id="results-output", className="mb-4"),
            dcc.Tabs([
                dcc.Tab(label="KVA Components", children=[dcc.Graph(id="kva-components-graph")]),
                dcc.Tab(label="Total XVA (Table 13.3-13.5)", children=[dcc.Graph(id="xva-total-graph")]),
            ])
        ], width=8)
    ])
], fluid=True)

@app.callback(
    [Output("results-output", "children"),
     Output("kva-components-graph", "figure"),
     Output("xva-total-graph", "figure")],
    [Input("calc-button", "n_clicks")],
    [State("notional", "value"),
     State("years", "value"),
     State("counterparty-rating", "value"),
     State("phi", "value"),
     State("gamma-k", "value"),
     State("issuer-spread", "value"),
     State("r-b", "value"),
     State("risk-free-rate", "value"),
     State("hedge-type", "value")]
)
def update_dashboard(n_clicks, notional, years, counterparty_rating, phi, gamma_k, issuer_spread, r_b, risk_free_rate, hedge_type):
    if n_clicks is None:
        return "Click Calculate to see results", go.Figure(), go.Figure()

    gamma_k = gamma_k / 100
    issuer_spread = issuer_spread / 100
    r_b = r_b / 100
    risk_free_rate = risk_free_rate / 100

    # KVA Calculation
    kva, k_mr, k_ccr, k_cva = kva_calc(notional, years, counterparty_rating, phi, gamma_k, issuer_spread, r_b, risk_free_rate, hedge_type)

    # Total XVA (from Tables 13.3-13.5)
    table_data = {
        "unhedged": {
            "AAA": {"CVA": -4, "DVA": 39, "FCA": -14, "KVA_0": -253, "KVA_1": -170, "Total_0": -253, "Total_1": -170},
            "A": {"CVA": -10, "DVA": 38, "FCA": -14, "KVA_0": -259, "KVA_1": -176, "Total_0": -259, "Total_1": -176},
            "BB": {"CVA": -31, "DVA": 33, "FCA": -12, "KVA_0": -279, "KVA_1": -198, "Total_0": -279, "Total_1": -198},
            "CCC": {"CVA": -68, "DVA": 24, "FCA": -9, "KVA_0": -341, "KVA_1": -260, "Total_0": -341, "Total_1": -260}
        },
        "back_to_back": {
            "AAA": {"CVA": -4, "DVA": 39, "FCA": -14, "KVA_0": 9, "KVA_1": 13, "Total_0": 9, "Total_1": 13},
            "A": {"CVA": -10, "DVA": 38, "FCA": -14, "KVA_0": -3, "KVA_1": 3, "Total_0": -3, "Total_1": 3},
            "BB": {"CVA": -31, "DVA": 33, "FCA": -12, "KVA_0": -45, "KVA_1": -32, "Total_0": -45, "Total_1": -32},
            "CCC": {"CVA": -68, "DVA": 24, "FCA": -9, "KVA_0": -156, "KVA_1": -125, "Total_0": -156, "Total_1": -125}
        },
        "ir01_hedged": {
            "AAA": {"CVA": -4, "DVA": 39, "FCA": -14, "KVA_0": -13, "KVA_1": -1, "Total_0": -13, "Total_1": -1},
            "A": {"CVA": -10, "DVA": 38, "FCA": -14, "KVA_0": -30, "KVA_1": -14, "Total_0": -30, "Total_1": -14},
            "BB": {"CVA": -31, "DVA": 33, "FCA": -12, "KVA_0": -88, "KVA_1": -59, "Total_0": -88, "Total_1": -59},
            "CCC": {"CVA": -68, "DVA": 24, "FCA": -9, "KVA_0": -249, "KVA_1": -187, "Total_0": -249, "Total_1": -187}
        }
    }

    data = table_data[hedge_type][counterparty_rating]
    cva = data["CVA"]
    dva = data["DVA"]
    fca = data["FCA"]
    kva_0 = data["KVA_0"]
    kva_1 = data["KVA_1"]
    total_0 = data["Total_0"]
    total_1 = data["Total_1"]

    # Results
    results = html.Div([
        html.P(f"KVA (Calculated, φ={phi}): {kva:.2f} bps"),
        html.P(f"KVA (Table, φ=0): {kva_0:.2f} bps"),
        html.P(f"KVA (Table, φ=1): {kva_1:.2f} bps"),
        html.P(f"Total XVA (Table, φ=0): {total_0:.2f} bps"),
        html.P(f"Total XVA (Table, φ=1): {total_1:.2f} bps"),
    ])

    # Graphs
    # KVA Components
    kva_components_fig = go.Figure(data=[
        go.Bar(name="Market Risk (MR)", x=["KVA Components"], y=[k_mr]),
        go.Bar(name="CCR", x=["KVA Components"], y=[k_ccr]),
        go.Bar(name="CVA", x=["KVA Components"], y=[k_cva]),
        go.Bar(name="Total KVA", x=["KVA Components"], y=[kva])
    ])
    kva_components_fig.update_layout(
        title=f"KVA Components (φ={phi}, {hedge_type})",
        yaxis_title="KVA (bps of Notional)",
        annotations=[dict(x=0, y=-50, text="Note: Breakdown of KVA. Interpretation: MR dominates for unhedged swaps.", showarrow=False, yshift=-50)]
    )

    # Total XVA
    xva_total_fig = go.Figure(data=[
        go.Bar(name="CVA", x=["XVA Terms"], y=[cva]),
        go.Bar(name="DVA", x=["XVA Terms"], y=[dva]),
        go.Bar(name="FCA", x=["XVA Terms"], y=[fca]),
        go.Bar(name="KVA (φ=0)", x=["XVA Terms"], y=[kva_0]),
        go.Bar(name="KVA (φ=1)", x=["XVA Terms"], y=[kva_1]),
        go.Bar(name="Total (φ=0)", x=["XVA Terms"], y=[total_0]),
        go.Bar(name="Total (φ=1)", x=["XVA Terms"], y=[total_1])
    ])
    xva_total_fig.update_layout(
        title=f"Total XVA (Table 13.3-13.5, {hedge_type}, {counterparty_rating})",
        yaxis_title="XVA (bps of Notional)",
        annotations=[dict(x=0, y=-100, text="Note: From Tables 13.3-13.5. Interpretation: KVA significant across scenarios.", showarrow=False, yshift=-50)]
    )

    return results, kva_components_fig, xva_total_fig

if __name__ == "__main__":
    app.run(debug=True)