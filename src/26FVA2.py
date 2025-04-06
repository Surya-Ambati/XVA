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
            html.H3("FVA Analysis Controls"),
            dcc.Tabs([
                # Exposure Profiles Tab
                dcc.Tab(label="Exposure Profiles", children=[
                    html.H5("Swap Parameters"),
                    dcc.Input(id='notional', value=100, type='number', placeholder="Notional ($M)"),
                    dcc.Input(id='coupon', value=5.0, type='number', placeholder="Coupon (%)"),
                    dcc.Input(id='maturity', value=10, type='number', placeholder="Maturity (years)"),
                    
                    html.H5("Market Data"),
                    dcc.Input(id='volatility', value=0.2, type='number', placeholder="Volatility"),
                    dcc.Input(id='mean-reversion', value=0.05, type='number', placeholder="Mean Reversion"),
                    
                    html.Button('Calculate Exposures', id='calc-exposures'),
                ]),
                
                # FVA Calculation Tab
                dcc.Tab(label="FVA Calculation", children=[
                    html.H5("Funding Parameters"),
                    dcc.Input(id='funding-spread', value=0.02, type='number', placeholder="Funding Spread (%)"),
                    dcc.Input(id='recovery-rate', value=0.4, type='number', placeholder="Recovery Rate"),
                    
                    html.H5("Collateral Parameters"),
                    dcc.Dropdown(
                        id='csa-type',
                        options=[
                            {'label': 'No CSA', 'value': 'none'},
                            {'label': 'Full CSA', 'value': 'full'},
                            {'label': 'Partial CSA', 'value': 'partial'}
                        ],
                        value='none'
                    ),
                    dcc.Input(id='threshold', value=10, type='number', placeholder="Threshold ($M)"),
                    
                    html.Button('Calculate FVA', id='calc-fva'),
                ]),
                
                # Hedging Analysis Tab
                dcc.Tab(label="Hedging Analysis", children=[
                    html.H5("Hedge Strategy"),
                    dcc.Dropdown(
                        id='hedge-type',
                        options=[
                            {'label': 'Back-to-back CSA', 'value': 'backtoback'},
                            {'label': 'Unsecured Hedge', 'value': 'unsecured'},
                            {'label': 'Partial Hedge', 'value': 'partial'}
                        ],
                        value='backtoback'
                    ),
                    
                    html.H5("Rehypothecation"),
                    dcc.RadioItems(
                        id='rehypothecation',
                        options=[
                            {'label': 'Allowed', 'value': True},
                            {'label': 'Not Allowed', 'value': False}
                        ],
                        value=True
                    ),
                    
                    html.Button('Analyze Hedge', id='analyze-hedge'),
                ])
            ]),
        ], width=4),
        
        # Results Panel
        dbc.Col([
            html.Div(id='exposure-output'),
            html.Div(id='fva-output'),
            html.Div(id='hedge-output'),
        ], width=8)
    ])
])

# Helper functions
def calculate_exposures(notional, coupon, maturity, volatility, mean_reversion):
    """Calculate expected exposure profiles using simple model"""
    time_points = np.linspace(0, maturity, 50)
    
    # Simplified exposure model (could be replaced with more sophisticated calculation)
    epe = notional * norm.pdf(norm.ppf(coupon/100)) * np.sqrt(time_points) * volatility * np.exp(-mean_reversion*time_points)
    ene = -notional * norm.pdf(norm.ppf(coupon/100)) * np.sqrt(time_points) * volatility * np.exp(-mean_reversion*time_points)
    
    return pd.DataFrame({'Time': time_points, 'EPE': epe, 'ENE': ene})

def calculate_fva(exposure_df, funding_spread, recovery_rate, csa_type, threshold):
    """Calculate FVA based on exposure profiles"""
    if csa_type == 'full':
        return 0.0, 0.0  # Return tuple of zeros for fully collateralized trades
    
    # Discount factors (simplified)
    discount_factors = np.exp(-0.02 * exposure_df['Time'])
    
    if csa_type == 'partial':
        # Apply threshold
        epe = np.maximum(exposure_df['EPE'] - threshold, 0)
        ene = np.maximum(exposure_df['ENE'] - threshold, 0)
    else:
        epe = exposure_df['EPE']
        ene = exposure_df['ENE']
    
    # FVA calculation
    fva_cost = np.sum(epe * funding_spread/100 * discount_factors * np.gradient(exposure_df['Time']))
    fva_benefit = np.sum(ene * funding_spread/100 * (1-recovery_rate) * discount_factors * np.gradient(exposure_df['Time']))
    
    return fva_cost, fva_benefit

def analyze_hedge_strategy(hedge_type, rehypothecation, exposure_df):
    """Analyze hedge strategy funding implications"""
    results = {}
    
    if hedge_type == 'backtoback':
        if rehypothecation:
            results['funding_cost'] = 0
            results['funding_benefit'] = 0
            results['description'] = "Perfect collateral match with rehypothecation - no funding impact"
        else:
            results['funding_cost'] = np.sum(exposure_df['EPE']) * 0.02  # Example calculation
            results['funding_benefit'] = np.sum(exposure_df['ENE']) * 0.02
            results['description'] = "Collateral needs to be sourced despite hedge"
    
    elif hedge_type == 'unsecured':
        results['funding_cost'] = np.sum(exposure_df['EPE']) * 0.05  # Higher cost for unsecured
        results['funding_benefit'] = np.sum(exposure_df['ENE']) * 0.03  # Lower benefit
        results['description'] = "Higher funding costs due to unsecured hedge"
    
    else:  # partial
        results['funding_cost'] = np.sum(exposure_df['EPE']) * 0.03
        results['funding_benefit'] = np.sum(exposure_df['ENE']) * 0.02
        results['description'] = "Partial hedge leaves residual funding exposure"
    
    return results

# Callbacks
@app.callback(
    Output('exposure-output', 'children'),
    Input('calc-exposures', 'n_clicks'),
    [State('notional', 'value'),
     State('coupon', 'value'),
     State('maturity', 'value'),
     State('volatility', 'value'),
     State('mean-reversion', 'value')]
)
def update_exposures(n_clicks, notional, coupon, maturity, volatility, mean_reversion):
    if n_clicks is None:
        return ""
    
    # Calculate exposures
    exposure_df = calculate_exposures(notional, coupon, maturity, volatility, mean_reversion)
    
    # Create exposure plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=exposure_df['Time'], y=exposure_df['EPE'], name='EPE', fill='tozeroy'))
    fig.add_trace(go.Scatter(x=exposure_df['Time'], y=exposure_df['ENE'], name='ENE', fill='tozeroy'))
    
    fig.update_layout(
        title='Expected Exposure Profiles',
        xaxis_title='Time (years)',
        yaxis_title='Exposure ($M)',
        hovermode='x unified'
    )
    
    # Create loan/deposit comparison
    loan_mtm = exposure_df['EPE'].max()
    deposit_mtm = exposure_df['ENE'].min()
    
    comp_fig = go.Figure()
    comp_fig.add_trace(go.Bar(
        x=['Swap EPE', 'Loan', 'Swap ENE', 'Deposit'],
        y=[exposure_df['EPE'].max(), loan_mtm, exposure_df['ENE'].min(), deposit_mtm],
        name='Comparison'
    ))
    
    comp_fig.update_layout(
        title='Exposure Comparison with Loan/Deposit',
        yaxis_title='Exposure ($M)'
    )
    
    notes = html.Div([
        html.H5("Exposure Profile Interpretation"),
        html.P("The Expected Positive Exposure (EPE) and Expected Negative Exposure (ENE) profiles show:"),
        html.Ul([
            html.Li("EPE: The average exposure when the bank is owed money (like a loan)"),
            html.Li("ENE: The average exposure when the bank owes money (like a deposit)"),
            html.Li("The shape depends on swap characteristics and market parameters")
        ]),
        html.P("Key observations:"),
        html.Ul([
            html.Li(f"Maximum EPE: ${exposure_df['EPE'].max():.2f}M (similar to loan)"),
            html.Li(f"Minimum ENE: ${exposure_df['ENE'].min():.2f}M (similar to deposit)"),
            html.Li("Derivatives can have both loan-like and deposit-like characteristics")
        ])
    ])
    
    return [
        dcc.Graph(figure=fig),
        dcc.Graph(figure=comp_fig),
        notes
    ]

@app.callback(
    Output('fva-output', 'children'),
    Input('calc-fva', 'n_clicks'),
    [State('notional', 'value'),
     State('coupon', 'value'),
     State('maturity', 'value'),
     State('volatility', 'value'),
     State('mean-reversion', 'value'),
     State('funding-spread', 'value'),
     State('recovery-rate', 'value'),
     State('csa-type', 'value'),
     State('threshold', 'value')]
)
def update_fva(n_clicks, notional, coupon, maturity, volatility, mean_reversion, 
              funding_spread, recovery_rate, csa_type, threshold):
    if n_clicks is None:
        return ""
    
    # Get exposures
    exposure_df = calculate_exposures(notional, coupon, maturity, volatility, mean_reversion)
    
    # Calculate FVA
    fva_cost, fva_benefit = calculate_fva(exposure_df, funding_spread, recovery_rate, csa_type, threshold)
    total_fva = fva_cost - fva_benefit
    
    # Create FVA breakdown
    fva_fig = go.Figure()
    fva_fig.add_trace(go.Bar(
        x=['FVA Cost', 'FVA Benefit', 'Net FVA'],
        y=[fva_cost, fva_benefit, total_fva],
        name='FVA Components'
    ))
    
    fva_fig.update_layout(
        title='FVA Calculation Breakdown',
        yaxis_title='FVA ($M)'
    )
    
    # CSA impact visualization
    csa_impact = {
        'none': total_fva,
        'partial': total_fva * 0.5,  # Simplified
        'full': 0
    }
    
    csa_fig = go.Figure()
    csa_fig.add_trace(go.Bar(
        x=['No CSA', 'Partial CSA', 'Full CSA'],
        y=[csa_impact['none'], csa_impact['partial'], csa_impact['full']],
        name='CSA Impact'
    ))
    
    csa_fig.update_layout(
        title='Impact of CSA Agreement on FVA',
        yaxis_title='FVA ($M)'
    )
    
    notes = html.Div([
        html.H5("FVA Interpretation"),
        html.P("Funding Valuation Adjustment reflects:"),
        html.Ul([
            html.Li(f"FVA Cost: ${fva_cost:.2f}M (funding EPE at bank's spread)"),
            html.Li(f"FVA Benefit: ${fva_benefit:.2f}M (benefit from ENE, reduced by recovery)"),
            html.Li(f"Net FVA: ${total_fva:.2f}M (impact on derivative valuation)")
        ]),
        html.P("CSA Impact:"),
        html.Ul([
            html.Li("No CSA: Full FVA impact"),
            html.Li("Partial CSA: Only exposures above threshold incur FVA"),
            html.Li("Full CSA: No FVA (funding at collateral rate)")
        ])
    ])
    
    return [
        dcc.Graph(figure=fva_fig),
        dcc.Graph(figure=csa_fig),
        notes
    ]

@app.callback(
    Output('hedge-output', 'children'),
    Input('analyze-hedge', 'n_clicks'),
    [State('notional', 'value'),
     State('coupon', 'value'),
     State('maturity', 'value'),
     State('volatility', 'value'),
     State('mean-reversion', 'value'),
     State('hedge-type', 'value'),
     State('rehypothecation', 'value')]
)
def update_hedge_analysis(n_clicks, notional, coupon, maturity, volatility, 
                         mean_reversion, hedge_type, rehypothecation):
    if n_clicks is None:
        return ""
    
    # Get exposures
    exposure_df = calculate_exposures(notional, coupon, maturity, volatility, mean_reversion)
    
    # Analyze hedge strategy
    results = analyze_hedge_strategy(hedge_type, rehypothecation, exposure_df)
    
    # Create hedge strategy comparison
    strategies = ['Back-to-back CSA', 'Unsecured Hedge', 'Partial Hedge']
    costs = [0.02 * np.sum(exposure_df['EPE']), 0.05 * np.sum(exposure_df['EPE']), 0.03 * np.sum(exposure_df['EPE'])]
    benefits = [0.02 * np.sum(exposure_df['ENE']), 0.03 * np.sum(exposure_df['ENE']), 0.02 * np.sum(exposure_df['ENE'])]
    
    hedge_fig = go.Figure()
    hedge_fig.add_trace(go.Bar(
        x=strategies,
        y=costs,
        name='Funding Cost'
    ))
    hedge_fig.add_trace(go.Bar(
        x=strategies,
        y=benefits,
        name='Funding Benefit'
    ))
    
    hedge_fig.update_layout(
        title='Hedge Strategy Comparison',
        yaxis_title='Funding Impact ($M)',
        barmode='group'
    )
    
    # Collateral flows diagram
    if hedge_type == 'backtoback' and rehypothecation:
        flow_fig = go.Figure(go.Sankey(
            node=dict(
                label=["Corporate", "Bank", "Market Hedge"],
                color=["blue", "green", "orange"]
            ),
            link=dict(
                source=[0, 1],
                target=[1, 2],
                value=[exposure_df['EPE'].max(), exposure_df['EPE'].max()]
            )
        ))
        flow_fig.update_layout(title='Collateral Flows with Rehypothecation')
    else:
        flow_fig = go.Figure(go.Sankey(
            node=dict(
                label=["Corporate", "Bank", "Market Hedge", "Money Market"],
                color=["blue", "green", "orange", "red"]
            ),
            link=dict(
                source=[0, 1, 1],
                target=[1, 2, 3],
                value=[0, exposure_df['EPE'].max(), exposure_df['EPE'].max()]
            )
        ))
        flow_fig.update_layout(title='Collateral Flows with Funding')
    
    notes = html.Div([
        html.H5("Hedge Strategy Analysis"),
        html.P(results['description']),
        html.P("Key findings:"),
        html.Ul([
            html.Li(f"Funding Cost: ${results['funding_cost']:.2f}M"),
            html.Li(f"Funding Benefit: ${results['funding_benefit']:.2f}M"),
            html.Li("Rehypothecation significantly reduces funding needs when allowed")
        ]),
        html.P("Optimal strategy depends on:"),
        html.Ul([
            html.Li("Counterparty credit quality"),
            html.Li("Available collateral"),
            html.Li("Market liquidity conditions")
        ])
    ])
    
    return [
        dcc.Graph(figure=hedge_fig),
        dcc.Graph(figure=flow_fig),
        notes
    ]

if __name__ == '__main__':
    app.run(debug=True)