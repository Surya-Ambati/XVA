import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Funding Curve Calculation
def funding_curve(cds_spread, bond_yield, risk_free_rate, r_b, treasury_spread, method="cds"):
    if method == "cds":
        lambda_b = cds_spread / (1 - r_b)
        s_f = (1 - r_b) * lambda_b
    elif method == "bond":
        lambda_b = (bond_yield - risk_free_rate) / (1 - r_b)
        s_f = (1 - r_b) * lambda_b
    else:  # treasury
        s_f = treasury_spread
    return s_f

# Weighted Multi-Currency Funding Spread
def weighted_funding_spread(currencies, amounts, spreads):
    total_amount = sum(amounts)
    weighted_spread = sum(a * s for a, s in zip(amounts, spreads)) / total_amount
    return weighted_spread

# MVA with Weighted Funding Spread
def mva_weighted(initial_margin, funding_spread, years=12):
    times = np.linspace(0, years, int(years * 12) + 1)
    dt = times[1] - times[0]
    discount = np.exp(-0.02 * np.array(times))  # Simplified discount
    mva = -sum(funding_spread * im * d * dt for im, d in zip(initial_margin, discount)) / 1e6
    return mva

# Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        # Left Panel: Inputs
        dbc.Col([
            html.H3("Inputs", className="text-center"),
            dbc.Label("CDS Spread (bps)"),
            dcc.Input(id="cds-spread", type="number", value=100, className="mb-3"),
            dbc.Label("Bond Yield (%)"),
            dcc.Input(id="bond-yield", type="number", value=4, className="mb-3"),
            dbc.Label("Risk-Free Rate (%)"),
            dcc.Input(id="risk-free-rate", type="number", value=2, className="mb-3"),
            dbc.Label("Bank Recovery Rate (R_B, %)"),
            dcc.Input(id="r-b", type="number", value=40, className="mb-3"),
            dbc.Label("Treasury Spread (%)"),
            dcc.Input(id="treasury-spread", type="number", value=1.5, className="mb-3"),
            dbc.Label("Funding Curve Method"),
            dcc.Dropdown(id="method", options=[
                {"label": "CDS Spread", "value": "cds"},
                {"label": "Bond Spread", "value": "bond"},
                {"label": "Treasury Transfer Price", "value": "treasury"}
            ], value="cds", className="mb-3"),
            dbc.Label("Initial Margin ($M)"),
            dcc.Input(id="initial-margin", type="number", value=5, className="mb-3"),
            dbc.Label("USD Amount ($M)"),
            dcc.Input(id="usd-amount", type="number", value=5, className="mb-3"),
            dbc.Label("USD Funding Spread (%)"),
            dcc.Input(id="usd-spread", type="number", value=4, className="mb-3"),
            dbc.Label("EUR Amount (€M)"),
            dcc.Input(id="eur-amount", type="number", value=3, className="mb-3"),
            dbc.Label("EUR Funding Spread (%)"),
            dcc.Input(id="eur-spread", type="number", value=3, className="mb-3"),
            dbc.Button("Calculate", id="calc-button", color="primary", className="mt-3"),
        ], width=4, className="bg-light p-3"),

        # Right Panel: Outputs and Graphs
        dbc.Col([
            html.H3("Results", className="text-center"),
            html.Div(id="results-output", className="mb-4"),
            dcc.Tabs([
                dcc.Tab(label="Funding Spread Comparison", children=[dcc.Graph(id="funding-spread-graph")]),
                dcc.Tab(label="MVA with Weighted Funding", children=[dcc.Graph(id="mva-graph")]),
            ])
        ], width=8)
    ])
], fluid=True)

@app.callback(
    [Output("results-output", "children"),
     Output("funding-spread-graph", "figure"),
     Output("mva-graph", "figure")],
    [Input("calc-button", "n_clicks")],
    [State("cds-spread", "value"),
     State("bond-yield", "value"),
     State("risk-free-rate", "value"),
     State("r-b", "value"),
     State("treasury-spread", "value"),
     State("method", "value"),
     State("initial-margin", "value"),
     State("usd-amount", "value"),
     State("usd-spread", "value"),
     State("eur-amount", "value"),
     State("eur-spread", "value")]
)
def update_dashboard(n_clicks, cds_spread, bond_yield, risk_free_rate, r_b, treasury_spread, method, initial_margin, usd_amount, usd_spread, eur_amount, eur_spread):
    if n_clicks is None:
        return "Click Calculate to see results", go.Figure(), go.Figure()

    cds_spread = cds_spread / 100  # Convert bps to %
    bond_yield = bond_yield / 100
    risk_free_rate = risk_free_rate / 100
    r_b = r_b / 100
    treasury_spread = treasury_spread / 100
    usd_spread = usd_spread / 100
    eur_spread = eur_spread / 100

    # Funding Spreads
    s_f_cds = funding_curve(cds_spread, bond_yield, risk_free_rate, r_b, treasury_spread, method="cds")
    s_f_bond = funding_curve(cds_spread, bond_yield, risk_free_rate, r_b, treasury_spread, method="bond")
    s_f_treasury = funding_curve(cds_spread, bond_yield, risk_free_rate, r_b, treasury_spread, method="treasury")
    selected_s_f = funding_curve(cds_spread, bond_yield, risk_free_rate, r_b, treasury_spread, method=method)

    # Weighted Multi-Currency Funding Spread
    weighted_s_f = weighted_funding_spread(["USD", "EUR"], [usd_amount, eur_amount], [usd_spread, eur_spread])

    # MVA with Weighted Funding Spread
    times = np.linspace(0, 12, 12 * 12 + 1)
    initial_margin = [initial_margin * 1e6] * len(times)  # Constant IM for simplicity
    mva = mva_weighted(initial_margin, weighted_s_f)

    # Results
    results = html.Div([
        html.P(f"Selected Funding Spread ({method}): {selected_s_f*100:.2f}%"),
        html.P(f"Weighted Funding Spread (USD/EUR): {weighted_s_f*100:.2f}%"),
        html.P(f"MVA with Weighted Funding: ${mva:.2f}M"),
    ])

    # Graphs
    # Funding Spread Comparison
    funding_spread_fig = go.Figure(data=[
        go.Bar(name="CDS Spread", x=["Funding Spread"], y=[s_f_cds*100]),
        go.Bar(name="Bond Spread", x=["Funding Spread"], y=[s_f_bond*100]),
        go.Bar(name="Treasury Spread", x=["Funding Spread"], y=[s_f_treasury*100]),
        go.Bar(name="Weighted (USD/EUR)", x=["Funding Spread"], y=[weighted_s_f*100])
    ])
    funding_spread_fig.update_layout(
        title="Funding Spread Comparison (11.2)",
        yaxis_title="Spread (%)",
        annotations=[dict(x=0, y=-0.5, text="Note: Different internal sources. Interpretation: CDS and bond spreads higher than treasury.", showarrow=False, yshift=-50)]
    )

    # MVA with Weighted Funding
    mva_fig = go.Figure(data=[
        go.Bar(name="MVA", x=["MVA"], y=[mva])
    ])
    mva_fig.update_layout(
        title="MVA with Weighted Funding (11.4)",
        yaxis_title="MVA ($M)",
        annotations=[dict(x=0, y=-0.1, text="Note: Multi-currency funding. Interpretation: Reflects weighted cost of collateral.", showarrow=False, yshift=-50)]
    )

    return results, funding_spread_fig, mva_fig

if __name__ == "__main__":
    app.run(debug=True)