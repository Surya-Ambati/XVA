import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import norm

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            # Control Panel
            html.H3("Burgard-Kjaer FVA Model Controls"),
            dcc.Tabs([
                # Model Parameters Tab
                dcc.Tab(label="Model Parameters", children=[
                    html.H5("Credit Parameters"),
                    dcc.Input(id='lambda-b', value=0.05, type='number', placeholder="Bank Hazard Rate (λB)"),
                    dcc.Input(id='lambda-c', value=0.03, type='number', placeholder="Cpty Hazard Rate (λC)"),
                    dcc.Input(id='recovery-b', value=0.4, type='number', placeholder="Bank Recovery (RB)"),
                    dcc.Input(id='recovery-c', value=0.4, type='number', placeholder="Cpty Recovery (RC)"),
                    
                    html.H5("Funding Parameters"),
                    dcc.Input(id='funding-spread', value=0.02, type='number', placeholder="Funding Spread (sF)"),
                    dcc.Input(id='risk-free-rate', value=0.03, type='number', placeholder="Risk-Free Rate (r)"),
                    
                    html.H5("Hedging Strategy"),
                    dcc.Dropdown(
                        id='hedge-strategy',
                        options=[
                            {'label': 'Perfect Replication', 'value': 'perfect'},
                            {'label': 'Semi-Replication (No Shortfall)', 'value': 'semi_no_shortfall'},
                            {'label': 'One-Bond Strategy', 'value': 'one_bond'}
                        ],
                        value='one_bond'
                    ),
                    
                    html.H5("Close-Out Assumptions"),
                    dcc.Dropdown(
                        id='close-out',
                        options=[
                            {'label': 'Standard (V)', 'value': 'standard'},
                            {'label': 'Risky (V̂)', 'value': 'risky'},
                            {'label': 'Survivor Funding (VFB)', 'value': 'survivor'}
                        ],
                        value='standard'
                    ),
                    
                    html.Button('Calculate XVA', id='calculate-xva'),
                ]),
                
                # Trade Parameters Tab
                dcc.Tab(label="Trade Parameters", children=[
                    html.H5("Exposure Profile"),
                    dcc.Input(id='epe', value=20, type='number', placeholder="EPE ($M)"),
                    dcc.Input(id='ene', value=15, type='number', placeholder="ENE ($M)"),
                    dcc.Input(id='exposure-vol', value=0.3, type='number', placeholder="Exposure Volatility"),
                    
                    html.H5("Collateral"),
                    dcc.Input(id='collateral', value=0, type='number', placeholder="Collateral ($M)"),
                    dcc.Input(id='collateral-spread', value=0.01, type='number', placeholder="Collateral Spread (sX)"),
                    
                    html.Button('Update Exposures', id='update-exposures'),
                ])
            ]),
        ], width=4),
        
        # Results Panel
        dbc.Col([
            html.Div(id='xva-output'),
            html.Div(id='exposure-output'),
            html.Div(id='strategy-output'),
        ], width=8)
    ])
])

# Helper functions
def calculate_xva_components(strategy, close_out, lambda_b, lambda_c, rb, rc, sF, r, epe, ene, collateral, sX):
    """Calculate XVA components based on Burgard-Kjaer model"""
    time_points = np.linspace(0, 5, 20)
    discount_factors = np.exp(-r * time_points)
    
    # Adjust for funding spread based on strategy
    if strategy == 'one_bond':
        funding_discount = np.exp(-(r + sF) * time_points)
    else:
        funding_discount = np.exp(-(r + lambda_b) * time_points)
    
    # Calculate exposures (simplified model)
    exposure = epe * np.exp(-0.1 * time_points)  # Decaying EPE
    negative_exposure = -ene * np.exp(-0.1 * time_points)  # Decaying ENE
    
    # Initialize components
    cva = np.zeros_like(time_points)
    dva = np.zeros_like(time_points)
    fca = np.zeros_like(time_points)
    colva = np.zeros_like(time_points)
    
    # Calculate components based on strategy
    if strategy == 'perfect':
        # Perfect replication - no FCA
        cva = (1 - rc) * exposure * lambda_c * discount_factors
        dva = (1 - rb) * negative_exposure * lambda_b * discount_factors
        fca = np.zeros_like(time_points)
    elif strategy == 'semi_no_shortfall':
        # Semi-replication with no shortfall
        cva = (1 - rc) * exposure * lambda_c * funding_discount
        dva = (1 - rb) * negative_exposure * lambda_b * funding_discount
        fca = (1 - rb) * exposure * lambda_b * funding_discount
    elif strategy == 'one_bond':
        # One-bond strategy
        cva = (1 - rc) * exposure * lambda_c * funding_discount
        fca = sF * (exposure + negative_exposure) * funding_discount
    
    # COLVA calculation
    colva = sX * collateral * discount_factors
    
    # Integrate over time
    dt = np.gradient(time_points)
    total_cva = -np.sum(cva * dt)
    total_dva = -np.sum(dva * dt)
    total_fca = -np.sum(fca * dt)
    total_colva = -np.sum(colva * dt)
    
    # Adjust for close-out assumptions
    if close_out == 'risky':
        # Risky close-out adjustment
        total_cva *= 1.2  # Simplified adjustment
    elif close_out == 'survivor':
        # Survivor funding close-out
        total_cva *= 0.9
        total_fca *= 1.1
    
    return {
        'CVA': total_cva,
        'DVA': total_dva,
        'FCA': total_fca,
        'COLVA': total_colva,
        'time_points': time_points,
        'exposure': exposure,
        'negative_exposure': negative_exposure
    }

# Callbacks
@app.callback(
    Output('xva-output', 'children'),
    Input('calculate-xva', 'n_clicks'),
    [State('hedge-strategy', 'value'),
     State('close-out', 'value'),
     State('lambda-b', 'value'),
     State('lambda-c', 'value'),
     State('recovery-b', 'value'),
     State('recovery-c', 'value'),
     State('funding-spread', 'value'),
     State('risk-free-rate', 'value'),
     State('epe', 'value'),
     State('ene', 'value'),
     State('collateral', 'value'),
     State('collateral-spread', 'value')]
)
def update_xva(n_clicks, strategy, close_out, lambda_b, lambda_c, rb, rc, sF, r, epe, ene, collateral, sX):
    if n_clicks is None:
        return ""
    
    results = calculate_xva_components(
        strategy, close_out, lambda_b/100, lambda_c/100, rb/100, rc/100, 
        sF/100, r/100, epe, ene, collateral, sX/100
    )
    
    # Create XVA breakdown
    components = ['CVA', 'DVA', 'FCA', 'COLVA']
    values = [results['CVA'], results['DVA'], results['FCA'], results['COLVA']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=components,
        y=values,
        name='XVA Components'
    ))
    
    fig.update_layout(
        title='XVA Components Breakdown',
        yaxis_title='Value ($M)',
        hovermode='x'
    )
    
    # Calculate total XVA
    total_xva = results['CVA'] + results['DVA'] + results['FCA'] + results['COLVA']
    
    notes = html.Div([
        html.H5("Burgard-Kjaer Model Interpretation"),
        html.P(f"Total XVA Adjustment: ${total_xva:.2f}M"),
        html.P("Components:"),
        html.Ul([
            html.Li(f"CVA: ${results['CVA']:.2f}M (counterparty risk)"),
            html.Li(f"DVA: ${results['DVA']:.2f}M (own credit)"),
            html.Li(f"FCA: ${results['FCA']:.2f}M (funding costs)"),
            html.Li(f"COLVA: ${results['COLVA']:.2f}M (collateral valuation)")
        ]),
        html.P("Model characteristics:"),
        html.Ul([
            html.Li(f"Hedging strategy: {strategy.replace('_', ' ').title()}"),
            html.Li(f"Close-out assumption: {close_out.title()}"),
            html.Li(f"Bank funding spread: {sF}% (λB={lambda_b}%, RB={rb}%)"),
            html.Li(f"Counterparty risk: λC={lambda_c}%, RC={rc}%")
        ])
    ])
    
    return [
        dcc.Graph(figure=fig),
        notes
    ]

@app.callback(
    Output('exposure-output', 'children'),
    Input('update-exposures', 'n_clicks'),
    [State('epe', 'value'),
     State('ene', 'value'),
     State('exposure-vol', 'value')]
)
def update_exposures(n_clicks, epe, ene, exposure_vol):
    if n_clicks is None:
        return ""
    
    # Create exposure profiles
    time_points = np.linspace(0, 5, 20)
    exposure = epe * np.exp(-exposure_vol * time_points)
    negative_exposure = -ene * np.exp(-exposure_vol * time_points)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points, y=exposure, name='EPE', fill='tozeroy'
    ))
    fig.add_trace(go.Scatter(
        x=time_points, y=negative_exposure, name='ENE', fill='tozeroy'
    ))
    
    fig.update_layout(
        title='Exposure Profiles',
        xaxis_title='Time (years)',
        yaxis_title='Exposure ($M)',
        hovermode='x unified'
    )
    
    notes = html.Div([
        html.H5("Exposure Profile Interpretation"),
        html.P("Key metrics:"),
        html.Ul([
            html.Li(f"Initial EPE: ${epe}M"),
            html.Li(f"Initial ENE: ${ene}M"),
            html.Li(f"Exposure decay rate: {exposure_vol}/year")
        ]),
        html.P("These profiles represent the expected positive and negative exposures used in XVA calculations.")
    ])
    
    return [
        dcc.Graph(figure=fig),
        notes
    ]

@app.callback(
    Output('strategy-output', 'children'),
    Input('calculate-xva', 'n_clicks'),
    [State('hedge-strategy', 'value'),
     State('close-out', 'value')]
)
def update_strategy(n_clicks, strategy, close_out):
    if n_clicks is None:
        return ""
    
    # Create strategy explanation
    strategy_descriptions = {
        'perfect': {
            'title': 'Perfect Replication Strategy',
            'points': [
                "Uses two bonds with different seniorities",
                "Perfectly hedges own default risk",
                "No FCA term (funding costs fully hedged)",
                "Matches equations 9.50-9.51 in text"
            ]
        },
        'semi_no_shortfall': {
            'title': 'Semi-Replication (No Shortfall)',
            'points': [
                "Uses two bonds but doesn't monetize windfall",
                "Ensures no shortfall on default",
                "Results in FCA term (eq 9.57)",
                "DVA can be viewed as funding benefit",
                "Matches equations 9.52-9.53"
            ]
        },
        'one_bond': {
            'title': 'One-Bond Strategy',
            'points': [
                "Uses single bond for funding",
                "Doesn't hedge own default risk",
                "Leads to symmetric FVA (eq 9.67)",
                "Common practitioner approach",
                "Matches equations 9.60-9.61"
            ]
        }
    }
    
    close_out_descriptions = {
        'standard': "MB = MC = V (Standard close-out)",
        'risky': "MB = MC = V̂ (Risky close-out including XVA)",
        'survivor': "MB = VFB, MC = VFC (Survivor funding)"
    }
    
    strategy_info = strategy_descriptions.get(strategy, {})
    close_out_info = close_out_descriptions.get(close_out, "")
    
    notes = html.Div([
        html.H5(strategy_info.get('title', '')),
        html.P(close_out_info),
        html.P("Key characteristics:"),
        html.Ul([html.Li(point) for point in strategy_info.get('points', [])]),
        html.P("Close-out impact:"),
        html.Ul([
            html.Li("Standard: Simple but may underestimate risk"),
            html.Li("Risky: More conservative but complex to implement"),
            html.Li("Survivor: Realistic but requires funding-adjusted valuations")
        ])
    ])
    
    return notes

if __name__ == '__main__':
    app.run(debug=True)