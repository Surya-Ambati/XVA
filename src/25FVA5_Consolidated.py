import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import norm

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Helper Functions
def generate_exposure_data(notional, fixed_rate, market_rate, maturity, volatility=0.2, mean_reversion=0.05):
    """Generate EPE/ENE profiles for a swap."""
    times = np.linspace(0, maturity, int(maturity * 12 + 1))
    epe, ene = [], []
    for t in times:
        if fixed_rate > market_rate:
            epe.append(notional * (fixed_rate - market_rate) * (1 - t/maturity) * np.exp(-mean_reversion*t) * 1e6)
            ene.append(-notional * (fixed_rate - market_rate) * 0.1 * np.exp(-mean_reversion*t) * 1e6)
        else:
            epe.append(notional * (fixed_rate - market_rate) * 0.1 * np.exp(-mean_reversion*t) * 1e6)
            ene.append(-notional * (fixed_rate - market_rate) * (1 - t/maturity) * np.exp(-mean_reversion*t) * 1e6)
    return times, epe, ene

def calculate_xva(epe, ene, risk_free_rate, funding_spread, lambda_b, lambda_c, r_b, r_c, collateral, collateral_spread,
                  strategy, close_out, correlation, hedge_type, rehypothecation, years=12):
    """Calculate XVA components using the Burgard-Kjaer model, with WWR/RWR and hedge type adjustments."""
    times = np.linspace(0, years, len(epe))
    dt = times[1] - times[0]
    
    # Discount factors (Burgard-Kjaer approach)
    discount = np.exp(-(risk_free_rate + lambda_b + lambda_c) * np.array(times))
    funding_rate = risk_free_rate + funding_spread
    funding_discount = np.exp(-(funding_rate + lambda_c) * np.array(times))
    
    # WWR/RWR: Stochastic funding spread
    np.random.seed(42)
    exposure = np.array(epe) + np.array(ene)
    spread_shocks = np.random.normal(0, 0.005, len(times))
    correlated_shocks = correlation * (exposure / np.max(np.abs(exposure))) + np.sqrt(1 - correlation**2) * spread_shocks
    stochastic_spread = funding_spread + correlated_shocks
    
    # Adjust exposures for collateral
    epe_adj = np.maximum(np.array(epe) - collateral * 1e6, 0)
    ene_adj = np.minimum(np.array(ene) + collateral * 1e6, 0)
    
    # Burgard-Kjaer Model Calculations
    # CVA and DVA (same for all strategies)
    cva = -(1 - r_c) * sum(lambda_c * d * max(e, 0) * dt for e, d in zip(epe_adj, discount)) / 1e6
    dva = -(1 - r_b) * sum(lambda_b * d * min(e, 0) * dt for e, d in zip(ene_adj, discount)) / 1e6
    
    # FCA/FVA depends on strategy
    if strategy == "perfect":
        fca = 0  # Perfect replication
    elif strategy == "semi_no_shortfall":
        fca = -(1 - r_b) * sum(lambda_b * d * max(e, 0) * dt for e, d in zip(epe_adj, discount)) / 1e6
    else:  # One-bond
        # Symmetric FVA with stochastic spread (WWR/RWR)
        fva = -sum(s * d * (e + n) * dt for e, n, s, d in zip(epe_adj, ene_adj, stochastic_spread, funding_discount)) / 1e6
        fca = fva  # In one-bond strategy, FCA is the FVA term
    
    # COLVA (Collateral Valuation Adjustment)
    colva = -sum(collateral_spread * collateral * d * dt for d in discount) / 1e6
    
    # Apply close-out adjustments
    if close_out == 'risky':
        cva *= 1.2
        fca *= 1.2
    elif close_out == 'survivor':
        cva *= 0.9
        fca *= 1.1
    
    # Apply hedge type adjustment
    if hedge_type == 'backtoback':
        hedge_factor = 0.1 if rehypothecation else 0.5
    elif hedge_type == 'unsecured':
        hedge_factor = 1.5
    else:  # partial
        hedge_factor = 0.8
    fca *= hedge_factor
    
    # Total XVA
    if strategy == "one_bond":
        total_xva = cva + fca + colva  # DVA is typically not included in one-bond strategy
    else:
        total_xva = cva + dva + fca + colva
    
    return cva, dva, fca, colva, total_xva, times, stochastic_spread, hedge_factor

# Layout
app.layout = dbc.Container([
    dbc.Row([
        # Control Panel
        dbc.Col([
            html.H3("FVA and XVA Analysis"),
            dcc.Tabs([
                dcc.Tab(label="Trade Parameters", children=[
                    html.H5("Swap Parameters"),
                    dbc.Label("Notional ($M)"),
                    dcc.Input(id='notional', value=100, type='number', className="mb-3"),
                    dbc.Label("Fixed Rate (%)"),
                    dcc.Input(id='fixed-rate', value=5, type='number', className="mb-3"),
                    dbc.Label("Market Rate (%)"),
                    dcc.Input(id='market-rate', value=3, type='number', className="mb-3"),
                    dbc.Label("Maturity (years)"),
                    dcc.Input(id='maturity', value=12, type='number', className="mb-3"),
                    dbc.Label("Volatility"),
                    dcc.Input(id='volatility', value=0.2, type='number', className="mb-3"),
                    dbc.Label("Mean Reversion"),
                    dcc.Input(id='mean-reversion', value=0.05, type='number', className="mb-3"),
                ]),
                dcc.Tab(label="Funding & Credit", children=[
                    html.H5("Funding Parameters"),
                    dbc.Label("Risk-Free Rate (%)"),
                    dcc.Input(id='risk-free-rate', value=2, type='number', className="mb-3"),
                    dbc.Label("Bank Funding Spread (%)"),
                    dcc.Input(id='funding-spread', value=1, type='number', className="mb-3"),
                    dbc.Label("Corporate Funding Spread (%)"),
                    dcc.Input(id='corp-spread', value=2, type='number', className="mb-3"),
                    dbc.Label("Correlation (WWR/RWR, -1 to 1)"),
                    dcc.Input(id='correlation', value=0.5, type='number', className="mb-3"),
                    html.H5("Credit Parameters"),
                    dbc.Label("Bank Default Intensity (λ_B, %)"),
                    dcc.Input(id='lambda-b', value=2, type='number', className="mb-3"),
                    dbc.Label("Counterparty Default Intensity (λ_C, %)"),
                    dcc.Input(id='lambda-c', value=1, type='number', className="mb-3"),
                    dbc.Label("Bank Recovery Rate (R_B, %)"),
                    dcc.Input(id='r-b', value=40, type='number', className="mb-3"),
                    dbc.Label("Counterparty Recovery Rate (R_C, %)"),
                    dcc.Input(id='r-c', value=40, type='number', className="mb-3"),
                ]),
                dcc.Tab(label="Collateral & Strategy", children=[
                    html.H5("Collateral Parameters"),
                    dbc.Label("Collateral ($M)"),
                    dcc.Input(id='collateral', value=0, type='number', className="mb-3"),
                    dbc.Label("Collateral Spread (%)"),
                    dcc.Input(id='collateral-spread', value=0.01, type='number', className="mb-3"),
                    dbc.Label("CSA Type"),
                    dcc.Dropdown(id='csa-type', options=[
                        {'label': 'No CSA', 'value': 'none'},
                        {'label': 'Full CSA', 'value': 'full'},
                        {'label': 'Partial CSA', 'value': 'partial'}
                    ], value='partial', className="mb-3"),
                    dbc.Label("Threshold ($M)"),
                    dcc.Input(id='threshold', value=10, type='number', className="mb-3"),
                    html.H5("Strategy Parameters"),
                    dbc.Label("Hedging Strategy"),
                    dcc.Dropdown(id='strategy', options=[
                        {'label': 'Perfect Replication', 'value': 'perfect'},
                        {'label': 'Semi-Replication (No Shortfall)', 'value': 'semi_no_shortfall'},
                        {'label': 'One-Bond', 'value': 'one_bond'}
                    ], value='one_bond', className="mb-3"),
                    dbc.Label("Close-Out Assumption"),
                    dcc.Dropdown(id='close-out', options=[
                        {'label': 'Standard', 'value': 'standard'},
                        {'label': 'Risky', 'value': 'risky'},
                        {'label': 'Survivor', 'value': 'survivor'}
                    ], value='standard', className="mb-3"),
                    dbc.Label("Hedge Type"),
                    dcc.Dropdown(id='hedge-type', options=[
                        {'label': 'Back-to-back CSA', 'value': 'backtoback'},
                        {'label': 'Unsecured Hedge', 'value': 'unsecured'},
                        {'label': 'Partial Hedge', 'value': 'partial'}
                    ], value='backtoback', className="mb-3"),
                    dbc.Label("Rehypothecation"),
                    dcc.RadioItems(id='rehypothecation', options=[
                        {'label': 'Allowed', 'value': True},
                        {'label': 'Not Allowed', 'value': False}
                    ], value=False, className="mb-3"),
                ])
            ]),
            # Calculate Button (Outside Tabs)
            dbc.Button("Calculate", id="calc-button", color="primary", className="mt-3"),
        ], width=4),
        
        # Results Panel
        dbc.Col([
            html.H3("Results"),
            html.Div(id='results-output'),
            dcc.Tabs([
                dcc.Tab(label="Exposure Profile", children=[dcc.Graph(id='exposure-graph')]),
                dcc.Tab(label="XVA Components", children=[dcc.Graph(id='xva-components-graph')]),
                dcc.Tab(label="WWR/RWR Impact", children=[dcc.Graph(id='wwr-graph')]),
            ])
        ], width=8)
    ])
], fluid=True)

# Callback to Update Collateral Input Dynamically
@app.callback(
    Output('collateral', 'value'),
    [Input('csa-type', 'value'),
     Input('notional', 'value'),
     Input('threshold', 'value')]
)
def update_collateral(csa_type, notional, threshold):
    if csa_type == 'full':
        return max(notional if notional is not None else 0, 0)  # Full CSA: collateral matches notional
    elif csa_type == 'partial':
        return min(0 if threshold is None else threshold, 0)  # Partial CSA: use threshold as cap, default to 0
    else:
        return 0  # No CSA: no collateral

# Callback to Update Dashboard
@app.callback(
    [Output('results-output', 'children'),
     Output('exposure-graph', 'figure'),
     Output('xva-components-graph', 'figure'),
     Output('wwr-graph', 'figure')],
    [Input('calc-button', 'n_clicks')],
    [State('notional', 'value'),
     State('fixed-rate', 'value'),
     State('market-rate', 'value'),
     State('maturity', 'value'),
     State('volatility', 'value'),
     State('mean-reversion', 'value'),
     State('risk-free-rate', 'value'),
     State('funding-spread', 'value'),
     State('corp-spread', 'value'),
     State('correlation', 'value'),
     State('lambda-b', 'value'),
     State('lambda-c', 'value'),
     State('r-b', 'value'),
     State('r-c', 'value'),
     State('collateral', 'value'),
     State('collateral-spread', 'value'),
     State('csa-type', 'value'),
     State('threshold', 'value'),
     State('strategy', 'value'),
     State('close-out', 'value'),
     State('hedge-type', 'value'),
     State('rehypothecation', 'value')]
)
def update_dashboard(n_clicks, notional, fixed_rate, market_rate, maturity, volatility, mean_reversion,
                    risk_free_rate, funding_spread, corp_spread, correlation, lambda_b, lambda_c, r_b, r_c,
                    collateral, collateral_spread, csa_type, threshold, strategy, close_out, hedge_type, rehypothecation):
    if n_clicks is None:
        return "Click Calculate to see results", go.Figure(), go.Figure(), go.Figure()
    
    # Convert percentages to decimals
    fixed_rate = (fixed_rate or 0) / 100
    market_rate = (market_rate or 0) / 100
    risk_free_rate = (risk_free_rate or 0) / 100
    funding_spread = (funding_spread or 0) / 100
    corp_spread = (corp_spread or 0) / 100
    lambda_b = (lambda_b or 0) / 100
    lambda_c = (lambda_c or 0) / 100
    r_b = (r_b or 0) / 100
    r_c = (r_c or 0) / 100
    collateral_spread = (collateral_spread or 0) / 100
    
    # Apply CSA adjustments (already reflected in UI via update_collateral callback)
    # For clarity, ensure collateral aligns with CSA type
    if csa_type == 'full':
        collateral = max(notional if notional is not None else 0, collateral or 0)
    elif csa_type == 'partial':
        collateral = min(collateral or 0, threshold if threshold is not None else 0)
    else:
        collateral = 0
    
    # Generate exposures
    times, epe, ene = generate_exposure_data(notional, fixed_rate, market_rate, maturity, volatility, mean_reversion)
    
    # Calculate XVA using Burgard-Kjaer model
    cva, dva, fca, colva, total_xva, times, stochastic_spread, hedge_factor = calculate_xva(
        epe, ene, risk_free_rate, funding_spread, lambda_b, lambda_c, r_b, r_c, collateral, collateral_spread,
        strategy, close_out, correlation, hedge_type, rehypothecation, years=maturity
    )
    
    # Results with Hedge Type Impact
    results = html.Div([
        html.P(f"CVA: ${cva:.2f}M"),
        html.P(f"DVA: ${dva:.2f}M"),
        html.P(f"FCA/FVA: ${fca:.2f}M"),
        html.P(f"COLVA: ${colva:.2f}M"),
        html.P(f"Total XVA: ${total_xva:.2f}M"),
        html.Hr(),
        html.P(f"Model: Burgard-Kjaer with {strategy.replace('_', ' ').title()} strategy and {close_out.title()} close-out."),
        html.P(f"Collateral Used: ${collateral:.2f}M (adjusted for {csa_type.replace('full', 'Full CSA').replace('partial', 'Partial CSA').replace('none', 'No CSA')})"),
        html.P(f"Hedge Type Impact: FCA adjusted by factor {hedge_factor:.2f} due to {hedge_type.replace('backtoback', 'Back-to-back CSA')} "
               f"with rehypothecation {'allowed' if rehypothecation else 'not allowed'}.")
    ])
    
    # Exposure Graph
    exposure_fig = go.Figure(data=[
        go.Scatter(x=times, y=epe, mode='lines', name='EPE'),
        go.Scatter(x=times, y=ene, mode='lines', name='ENE')
    ])
    exposure_fig.update_layout(
        title="Exposure Profile",
        xaxis_title="Time (Years)",
        yaxis_title="Exposure ($M)",
        annotations=[dict(x=0, y=-5, text="EPE/ENE drive XVA calculations.", showarrow=False, yshift=-50)]
    )
    
    # XVA Components Graph
    xva_fig = go.Figure(data=[
        go.Bar(name="CVA", x=["XVA"], y=[cva]),
        go.Bar(name="DVA", x=["XVA"], y=[dva]),
        go.Bar(name="FCA/FVA", x=["XVA"], y=[fca]),
        go.Bar(name="COLVA", x=["XVA"], y=[colva]),
        go.Bar(name="Total XVA", x=["XVA"], y=[total_xva])
    ])
    xva_fig.update_layout(
        title=f"XVA Components ({strategy.replace('_', ' ').title()} Strategy)",
        yaxis_title="Adjustment ($M)",
        annotations=[dict(x=0, y=-5, text=f"Strategy: {strategy}, Close-Out: {close_out}, Hedge: {hedge_type.replace('backtoback', 'Back-to-back CSA')}", showarrow=False, yshift=-50)]
    )
    
    # WWR/RWR Graph
    wwr_fig = go.Figure(data=[
        go.Scatter(x=times, y=epe, mode='lines', name='EPE'),
        go.Scatter(x=times, y=[s * 100 for s in stochastic_spread], mode='lines', name='Stochastic Spread (%)', yaxis="y2")
    ])
    wwr_fig.update_layout(
        title="WWR/RWR Impact on FVA",
        xaxis_title="Time (Years)",
        yaxis_title="Exposure ($M)",
        yaxis2=dict(title="Funding Spread (%)", overlaying="y", side="right"),
        annotations=[dict(x=0, y=-5, text=f"Correlation = {correlation}", showarrow=False, yshift=-50)]
    )
    
    return results, exposure_fig, xva_fig, wwr_fig

if __name__ == '__main__':
    app.run(debug=True)