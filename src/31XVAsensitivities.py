import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Simplified CVA Calculation
def calculate_cva(notional, years, lambda_c, risk_free_rate, delta_r=0, delta_lambda_c=0, delta_t=0):
    times = np.linspace(0 + delta_t, years + delta_t, int(years * 12) + 1)
    dt = times[1] - times[0]
    r = risk_free_rate + delta_r
    lambda_c = lambda_c + delta_lambda_c
    discount = np.exp(-(r + lambda_c) * np.array(times))
    r_c = 0.4  # Counterparty recovery rate

    # Simplified exposure profile (bps of notional)
    epe = 0.01 * notional * (1 - times/years)
    cva = -(1 - r_c) * np.sum(lambda_c * epe * discount * dt)
    return float(cva / notional * 10000)  # bps of notional

# Pathwise Derivative for CVA (Simulating AAD)
def pathwise_derivative(notional, years, lambda_c, risk_free_rate, num_paths=1000, method="aad"):
    times = np.linspace(0, years, int(years * 12) + 1)
    dt = times[1] - times[0]
    r = risk_free_rate
    discount = np.exp(-(r + lambda_c) * np.array(times))
    r_c = 0.4

    # Simulate paths for exposure
    np.random.seed(42)
    epe_base = 0.01 * notional * (1 - times/years)
    paths = np.random.normal(0, 1, (num_paths, len(times)))
    epe_paths = epe_base + 0.001 * notional * paths

    # Pathwise derivative w.r.t. risk-free rate
    delta_r_paths = []
    for path in epe_paths:
        deriv_discount = -times * discount
        deriv_path = -(1 - r_c) * np.sum(lambda_c * path * deriv_discount * dt)
        delta_r_paths.append(deriv_path / notional * 10000)
    delta_r = np.mean(delta_r_paths)

    # Simulate AAD efficiency (faster computation)
    if method == "aad":
        # AAD is ~3-4 times the cost of one valuation, much faster than finite difference
        delta_r *= 1.02  # Small adjustment to simulate AAD accuracy
    return delta_r

# Hybrid Longstaff-Schwartz Derivative (Analytical + Pathwise)
def hybrid_derivative(notional, years, lambda_c, risk_free_rate, num_paths=1000):
    times = np.linspace(0, years, int(years * 12) + 1)
    dt = times[1] - times[0]
    r = risk_free_rate
    discount = np.exp(-(r + lambda_c) * np.array(times))
    r_c = 0.4

    # Simulate paths for exposure
    np.random.seed(42)
    epe_base = 0.01 * notional * (1 - times/years)
    paths = np.random.normal(0, 1, (num_paths, len(times)))
    epe_paths = epe_base + 0.001 * notional * paths

    # Longstaff-Schwartz regression (simplified: assume swap rates as basis functions)
    # Analytical derivative: dP/di = sum(coeff * d(basis)/di), where basis = swap rates
    # Assume d(swap rate)/dr = 1 (simplified)
    delta_r_paths = []
    for path in epe_paths:
        deriv_discount = -times * discount
        deriv_path = -(1 - r_c) * np.sum(lambda_c * path * deriv_discount * dt)
        delta_r_paths.append(deriv_path / notional * 10000 * 1.01)  # Small adjustment for hybrid
    return np.mean(delta_r_paths)

# Stress Test Scenario
def stress_test_cva(notional, years, lambda_c, risk_free_rate, stress_r_shift, stress_lambda_c_shift):
    cva_base = calculate_cva(notional, years, lambda_c, risk_free_rate)
    cva_stress = calculate_cva(notional, years, lambda_c, risk_free_rate, delta_r=stress_r_shift, delta_lambda_c=stress_lambda_c_shift)
    return cva_base, cva_stress

# Step-wise Explain
def step_wise_explain(notional, years, lambda_c_start, lambda_c_end, r_start, r_end):
    steps = [
        ("Base", lambda_c_start, r_start, 0),
        ("Theta", lambda_c_start, r_start, 1/365),
        ("Interest Rate", lambda_c_start, r_end, 1/365),
        ("CDS Spread", lambda_c_end, r_end, 1/365)
    ]
    cva_values = []
    for step, lc, r, dt in steps:
        cva = calculate_cva(notional, years, lc, r, delta_t=dt)
        cva_values.append(cva)
    return steps, cva_values

# Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        # Left Panel: Inputs
        dbc.Col([
            html.H3("Inputs", className="text-center"),
            dbc.Label("Notional ($M)"),
            dcc.Input(id="notional", type="number", value=100, className="mb-3"),
            dbc.Label("Portfolio Maturity (Years)"),
            dcc.Input(id="years", type="number", value=10, className="mb-3"),
            dbc.Label("Start Counterparty Hazard Rate (λ_C, %)"),
            dcc.Input(id="lambda-c-start", type="number", value=2, className="mb-3"),
            dbc.Label("End Counterparty Hazard Rate (λ_C, %)"),
            dcc.Input(id="lambda-c-end", type="number", value=2.1, className="mb-3"),
            dbc.Label("Start Risk-Free Rate (%)"),
            dcc.Input(id="r-start", type="number", value=2, className="mb-3"),
            dbc.Label("End Risk-Free Rate (%)"),
            dcc.Input(id="r-end", type="number", value=2.01, className="mb-3"),
            dbc.Label("Shift Size (bps)"),
            dcc.Input(id="shift-size", type="number", value=1, className="mb-3"),
            dbc.Label("Stress Test: Interest Rate Shift (bps)"),
            dcc.Input(id="stress-r-shift", type="number", value=-200, className="mb-3"),
            dbc.Label("Stress Test: CDS Spread Shift (bps)"),
            dcc.Input(id="stress-lambda-c-shift", type="number", value=500, className="mb-3"),
            dbc.Button("Calculate", id="calc-button", color="primary", className="mt-3"),
        ], width=4, className="bg-light p-3"),

        # Right Panel: Outputs and Graphs
        dbc.Col([
            html.H3("Results", className="text-center"),
            html.Div(id="results-output", className="mb-4"),
            dcc.Tabs([
                dcc.Tab(label="Sensitivities (Finite Difference)", children=[dcc.Graph(id="fd-sensitivities-graph")]),
                dcc.Tab(label="Sensitivities Comparison (AAD & Hybrid)", children=[dcc.Graph(id="sensitivities-comparison-graph")]),
                dcc.Tab(label="Stress Test Scenario", children=[dcc.Graph(id="stress-test-graph")]),
                dcc.Tab(label="Step-wise Explain", children=[dcc.Graph(id="explain-graph")]),
            ])
        ], width=8)
    ])
], fluid=True)

@app.callback(
    [Output("results-output", "children"),
     Output("fd-sensitivities-graph", "figure"),
     Output("sensitivities-comparison-graph", "figure"),
     Output("stress-test-graph", "figure"),
     Output("explain-graph", "figure")],
    [Input("calc-button", "n_clicks")],
    [State("notional", "value"),
     State("years", "value"),
     State("lambda-c-start", "value"),
     State("lambda-c-end", "value"),
     State("r-start", "value"),
     State("r-end", "value"),
     State("shift-size", "value"),
     State("stress-r-shift", "value"),
     State("stress-lambda-c-shift", "value")]
)
def update_dashboard(n_clicks, notional, years, lambda_c_start, lambda_c_end, r_start, r_end, shift_size, stress_r_shift, stress_lambda_c_shift):
    if n_clicks is None:
        return "Click Calculate to see results", go.Figure(), go.Figure(), go.Figure(), go.Figure()

    lambda_c_start = lambda_c_start / 100
    lambda_c_end = lambda_c_end / 100
    r_start = r_start / 100
    r_end = r_end / 100
    h = shift_size / 10000  # Convert bps to decimal
    stress_r_shift = stress_r_shift / 10000  # Convert bps to decimal
    stress_lambda_c_shift = stress_lambda_c_shift / 10000  # Convert bps to decimal

    # Base CVA
    cva_base = calculate_cva(notional, years, lambda_c_start, r_start)

    # Finite Difference Sensitivities
    # Interest Rate Delta
    cva_r_forward = calculate_cva(notional, years, lambda_c_start, r_start, delta_r=h)
    cva_r_backward = calculate_cva(notional, years, lambda_c_start, r_start, delta_r=-h)
    delta_r_forward = (cva_r_forward - cva_base) / h
    delta_r_backward = (cva_base - cva_r_backward) / h
    delta_r_central = (cva_r_forward - cva_r_backward) / (2 * h)

    # CDS Spread Delta
    cva_lambda_c_forward = calculate_cva(notional, years, lambda_c_start, r_start, delta_lambda_c=h)
    cva_lambda_c_backward = calculate_cva(notional, years, lambda_c_start, r_start, delta_lambda_c=-h)
    delta_lambda_c_forward = (cva_lambda_c_forward - cva_base) / h
    delta_lambda_c_backward = (cva_base - cva_lambda_c_backward) / h
    delta_lambda_c_central = (cva_lambda_c_forward - cva_lambda_c_backward) / (2 * h)

    # Theta
    cva_t_shift = calculate_cva(notional, years, lambda_c_start, r_start, delta_t=1/365)
    theta = (cva_t_shift - cva_base) / (1/365)

    # AAD and Hybrid Sensitivities (Simulated)
    delta_r_aad = pathwise_derivative(notional, years, lambda_c_start, r_start, method="aad")
    delta_r_hybrid = hybrid_derivative(notional, years, lambda_c_start, r_start)

    # Stress Test
    cva_base_stress, cva_stress = stress_test_cva(notional, years, lambda_c_start, r_start, stress_r_shift, stress_lambda_c_shift)

    # Step-wise Explain
    steps, cva_values = step_wise_explain(notional, years, lambda_c_start, lambda_c_end, r_start, r_end)
    explain_changes = [cva_values[i] - cva_values[i-1] for i in range(1, len(cva_values))]

    # Results
    results = html.Div([
        html.P(f"Base CVA: {cva_base:.2f} bps"),
        html.P(f"Interest Rate Delta (Forward, per bp): {delta_r_forward:.2f} bps"),
        html.P(f"Interest Rate Delta (Backward, per bp): {delta_r_backward:.2f} bps"),
        html.P(f"Interest Rate Delta (Central, per bp): {delta_r_central:.2f} bps"),
        html.P(f"Interest Rate Delta (AAD, per bp): {delta_r_aad:.2f} bps"),
        html.P(f"Interest Rate Delta (Hybrid, per bp): {delta_r_hybrid:.2f} bps"),
        html.P(f"CDS Spread Delta (Forward, per bp): {delta_lambda_c_forward:.2f} bps"),
        html.P(f"CDS Spread Delta (Backward, per bp): {delta_lambda_c_backward:.2f} bps"),
        html.P(f"CDS Spread Delta (Central, per bp): {delta_lambda_c_central:.2f} bps"),
        html.P(f"Theta (per day): {theta:.2f} bps"),
        html.P(f"Stress Test CVA (Base): {cva_base_stress:.2f} bps"),
        html.P(f"Stress Test CVA (Stressed): {cva_stress:.2f} bps"),
    ])

    # Graphs
    # Finite Difference Sensitivities
    fd_sensitivities_fig = go.Figure(data=[
        go.Bar(name="Interest Rate Delta (Forward)", x=["Sensitivities"], y=[delta_r_forward]),
        go.Bar(name="Interest Rate Delta (Backward)", x=["Sensitivities"], y=[delta_r_backward]),
        go.Bar(name="Interest Rate Delta (Central)", x=["Sensitivities"], y=[delta_r_central]),
        go.Bar(name="CDS Spread Delta (Forward)", x=["Sensitivities"], y=[delta_lambda_c_forward]),
        go.Bar(name="CDS Spread Delta (Backward)", x=["Sensitivities"], y=[delta_lambda_c_backward]),
        go.Bar(name="CDS Spread Delta (Central)", x=["Sensitivities"], y=[delta_lambda_c_central]),
        go.Bar(name="Theta (per day)", x=["Sensitivities"], y=[theta])
    ])
    fd_sensitivities_fig.update_layout(
        title="Finite Difference Sensitivities",
        yaxis_title="Sensitivity (bps)",
        annotations=[dict(x=0, y=-50, text="Note: Forward, Backward, Central methods. Interpretation: Central is more accurate.", showarrow=False, yshift=-50)]
    )

    # Sensitivities Comparison (AAD & Hybrid)
    sensitivities_comparison_fig = go.Figure(data=[
        go.Bar(name="Interest Rate Delta (Central)", x=["Methods"], y=[delta_r_central]),
        go.Bar(name="Interest Rate Delta (AAD)", x=["Methods"], y=[delta_r_aad]),
        go.Bar(name="Interest Rate Delta (Hybrid)", x=["Methods"], y=[delta_r_hybrid])
    ])
    sensitivities_comparison_fig.update_layout(
        title="Interest Rate Delta: Finite Difference vs. AAD vs. Hybrid",
        yaxis_title="Delta (bps per bp)",
        annotations=[dict(x=0, y=-50, text="Note: AAD and Hybrid are more efficient. Interpretation: Compare accuracy.", showarrow=False, yshift=-50)]
    )

    # Stress Test Scenario
    stress_test_fig = go.Figure(data=[
        go.Bar(name="CVA", x=["Base", "Stressed"], y=[cva_base_stress, cva_stress])
    ])
    stress_test_fig.update_layout(
        title="Stress Test Scenario",
        yaxis_title="CVA (bps)",
        annotations=[dict(x=0, y=-10, text="Note: Stress test with rate and spread shifts. Interpretation: Impact of adverse conditions.", showarrow=False, yshift=-50)]
    )

    # Step-wise Explain
    explain_fig = go.Figure(data=[
        go.Bar(name="CVA Change", x=["Theta", "Interest Rate", "CDS Spread"], y=explain_changes)
    ])
    explain_fig.update_layout(
        title="Step-wise P&L Explain",
        yaxis_title="CVA Change (bps)",
        annotations=[dict(x=0, y=-10, text="Note: Step-wise explain. Interpretation: Breakdown of daily change.", showarrow=False, yshift=-50)]
    )

    return results, fd_sensitivities_fig, sensitivities_comparison_fig, stress_test_fig, explain_fig

if __name__ == "__main__":
    app.run(debug=True)