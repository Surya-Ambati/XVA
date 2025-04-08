import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.linear_model import LinearRegression

# Exposure Data with Longstaff-Schwartz Approximation
def longstaff_schwartz_exposure(notional, fixed_rate, market_rate, years=12, paths=1000):
    times = np.linspace(0, years, years * 12 + 1)
    np.random.seed(42)
    
    # Simulate explanatory variables (e.g., swap rate)
    swap_rates = np.random.normal(fixed_rate, 0.01, (len(times), paths))
    values = notional * (swap_rates - market_rate) * 1e6  # Simplified valuation
    
    # Longstaff-Schwartz regression (simplified)
    epe, ene = [], []
    for t in range(len(times)):
        # Use swap rate as explanatory variable, fit quadratic polynomial
        X = np.vstack([swap_rates[t], swap_rates[t]**2]).T
        y = values[t]
        reg = LinearRegression().fit(X, y)
        # Predict values across paths
        predicted_values = reg.predict(X)
        epe.append(np.mean(np.maximum(predicted_values, 0)))
        ene.append(np.mean(np.minimum(predicted_values, 0)))
    
    return times, epe, ene

# MVA with No Shortfall Strategy
def mva_no_shortfall(initial_margin, funding_spread, lambda_b, r_b, s_i, years=12):
    times = np.linspace(0, years, int(years * 12) + 1)
    dt = times[1] - times[0]
    discount = np.exp(-(0.02 + lambda_b) * np.array(times))  # Simplified discount (r + λ_B + λ_C)
    s_f = (1 - r_b) * lambda_b  # Funding spread
    mva = -sum((s_f - s_i) * initial_margin * d * dt for d in discount) / 1e6
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
            dbc.Label("Initial Margin ($M)"),
            dcc.Input(id="initial-margin", type="number", value=5, className="mb-3"),
            dbc.Label("Funding Spread (%)"),
            dcc.Input(id="funding-spread", type="number", value=1.2, className="mb-3"),
            dbc.Label("Bank Default Intensity (λ_B, %)"),
            dcc.Input(id="lambda-b", type="number", value=2, className="mb-3"),
            dbc.Label("Bank Recovery Rate (R_B, %)"),
            dcc.Input(id="r-b", type="number", value=40, className="mb-3"),
            dbc.Label("Initial Margin Spread (s_I, %)"),
            dcc.Input(id="s-i", type="number", value=0.5, className="mb-3"),
            dbc.Button("Calculate", id="calc-button", color="primary", className="mt-3"),
        ], width=4, className="bg-light p-3"),

        # Right Panel: Outputs and Graphs
        dbc.Col([
            html.H3("Results", className="text-center"),
            html.Div(id="results-output", className="mb-4"),
            dcc.Tabs([
                dcc.Tab(label="Exposure Profile (Longstaff-Schwartz)", children=[dcc.Graph(id="exposure-graph")]),
                dcc.Tab(label="MVA (No Shortfall Strategy)", children=[dcc.Graph(id="mva-graph")]),
            ])
        ], width=8)
    ])
], fluid=True)

@app.callback(
    [Output("results-output", "children"),
     Output("exposure-graph", "figure"),
     Output("mva-graph", "figure")],
    [Input("calc-button", "n_clicks")],
    [State("notional", "value"),
     State("fixed-rate", "value"),
     State("market-rate", "value"),
     State("initial-margin", "value"),
     State("funding-spread", "value"),
     State("lambda-b", "value"),
     State("r-b", "value"),
     State("s-i", "value")]
)
def update_dashboard(n_clicks, notional, fixed_rate, market_rate, initial_margin, funding_spread, lambda_b, r_b, s_i):
    if n_clicks is None:
        return "Click Calculate to see results", go.Figure(), go.Figure()

    fixed_rate = fixed_rate / 100
    market_rate = market_rate / 100
    funding_spread = funding_spread / 100
    lambda_b = lambda_b / 100
    r_b = r_b / 100
    s_i = s_i / 100

    # Exposure Profile with Longstaff-Schwartz
    times, epe, ene = longstaff_schwartz_exposure(notional, fixed_rate, market_rate, years=12)

    # MVA with No Shortfall Strategy
    mva = mva_no_shortfall(initial_margin * 1e6, funding_spread, lambda_b, r_b, s_i, years=12)

    # Results
    results = html.Div([
        html.P(f"MVA (No Shortfall Strategy): ${mva:.2f}M"),
    ])

    # Graphs
    # Exposure Profile
    exposure_fig = go.Figure(data=[
        go.Scatter(x=times, y=epe, mode='lines', name='EPE'),
        go.Scatter(x=times, y=ene, mode='lines', name='ENE')
    ])
    exposure_fig.update_layout(
        title="Exposure Profile (Longstaff-Schwartz, 10.3.2)",
        xaxis_title="Time (Years)",
        yaxis_title="Exposure ($M)",
        annotations=[dict(x=0, y=-5, text="Note: Longstaff-Schwartz approximation. Interpretation: Efficient exposure modeling.", showarrow=False, yshift=-50)]
    )

    # MVA
    mva_fig = go.Figure(data=[
        go.Bar(name="MVA", x=["MVA"], y=[mva])
    ])
    mva_fig.update_layout(
        title="MVA (No Shortfall Strategy, 10.2.1)",
        yaxis_title="MVA ($M)",
        annotations=[dict(x=0, y=-0.1, text="Note: Funding cost of IM. Interpretation: Reflects non-rehypothecatable margin.", showarrow=False, yshift=-50)]
    )

    return results, exposure_fig, mva_fig

if __name__ == "__main__":
    app.run(debug=True)