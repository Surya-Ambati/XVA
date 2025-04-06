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
            html.H3("FVA Model Analysis"),
            dcc.Tabs([
                # First Generation Models Tab
                dcc.Tab(label="Discount Models", children=[
                    html.H5("Market Data"),
                    dcc.Input(id='libor-rate', value=0.03, type='number', placeholder="3M LIBOR (%)"),
                    dcc.Input(id='funding-spread', value=0.02, type='number', placeholder="Funding Spread (%)"),
                    
                    html.H5("Trade Parameters"),
                    dcc.Input(id='notional', value=100, type='number', placeholder="Notional ($M)"),
                    dcc.Input(id='maturity', value=5, type='number', placeholder="Maturity (years)"),
                    
                    html.H5("Collateralization"),
                    dcc.Dropdown(
                        id='collateral-type',
                        options=[
                            {'label': 'Unsecured', 'value': 'unsecured'},
                            {'label': 'Partial CSA', 'value': 'partial'},
                            {'label': 'Full CSA', 'value': 'full'}
                        ],
                        value='unsecured'
                    ),
                    
                    html.Button('Calculate FVA', id='calc-discount-fva'),
                ]),
                
                # Double Counting Tab
                dcc.Tab(label="DVA Double Counting", children=[
                    html.H5("Credit Parameters"),
                    dcc.Input(id='default-intensity', value=0.05, type='number', placeholder="Default Intensity (λ)"),
                    dcc.Input(id='recovery-rate', value=0.4, type='number', placeholder="Recovery Rate"),
                    dcc.Input(id='bond-cds-basis', value=0.01, type='number', placeholder="Bond-CDS Basis (γ)"),
                    
                    html.H5("Counterparty Types"),
                    dcc.RadioItems(
                        id='counterparty-type',
                        options=[
                            {'label': 'Risky Borrower', 'value': 'risky_borrower'},
                            {'label': 'Risky Lender', 'value': 'risky_lender'}
                        ],
                        value='risky_borrower'
                    ),
                    
                    html.Button('Analyze Double Counting', id='analyze-double-counting'),
                ]),
                
                # Second Generation Models Tab
                dcc.Tab(label="Exposure Models", children=[
                    html.H5("Exposure Parameters"),
                    dcc.Input(id='epe', value=20, type='number', placeholder="EPE ($M)"),
                    dcc.Input(id='ene', value=15, type='number', placeholder="ENE ($M)"),
                    
                    html.H5("Funding Spread Term Structure"),
                    dcc.Input(id='short-term-spread', value=0.03, type='number', placeholder="Short-term (%)"),
                    dcc.Input(id='long-term-spread', value=0.02, type='number', placeholder="Long-term (%)"),
                    
                    html.Button('Calculate Exposure FVA', id='calc-exposure-fva'),
                ])
            ]),
        ], width=4),
        
        # Results Panel
        dbc.Col([
            html.Div(id='discount-model-output'),
            html.Div(id='double-counting-output'),
            html.Div(id='exposure-model-output'),
        ], width=8)
    ])
])

# Helper functions
def calculate_discount_fva(libor_rate, funding_spread, notional, maturity, collateral_type):
    """Calculate first-generation FVA using discount adjustment"""
    risk_free_rate = libor_rate / 100
    spread = funding_spread / 100
    
    if collateral_type == 'full':
        adj_rate = risk_free_rate  # OIS discounting for full CSA
    elif collateral_type == 'partial':
        adj_rate = risk_free_rate + spread * 0.5  # Partial adjustment
    else:
        adj_rate = risk_free_rate + spread  # Full funding adjustment
    
    # Simple PV calculation
    pv_risk_free = notional * np.exp(-risk_free_rate * maturity)
    pv_adjusted = notional * np.exp(-adj_rate * maturity)
    fva = pv_adjusted - pv_risk_free
    
    return pv_risk_free, pv_adjusted, fva

def analyze_double_counting(default_intensity, recovery_rate, bond_cds_basis, counterparty_type):
    """Analyze DVA/FVA double counting issue"""
    lambda_b = default_intensity / 100
    gamma_b = bond_cds_basis / 100
    R = recovery_rate / 100
    
    if counterparty_type == 'risky_borrower':
        # Lender's CVA
        cva = (1 - np.exp(-lambda_b * 5))  # Using 5y maturity as example
        # Borrower's FVA-adjusted value
        fva_adj = (1 - np.exp(-(2*lambda_b + gamma_b) * 5))
        # Traditional DVA
        dva = (1 - np.exp(-lambda_b * 5))
    else:
        # Lender's FVA
        cva = 0
        fva_adj = (1 - np.exp(-(lambda_b + gamma_b) * 5))
        dva = 0
    
    return cva, fva_adj, dva

def calculate_exposure_fva(epe, ene, short_term_spread, long_term_spread):
    """Calculate second-generation exposure-based FVA"""
    short_spread = short_term_spread / 100
    long_spread = long_term_spread / 100
    
    # Simple linear term structure
    def spread_curve(t):
        return short_spread + (long_spread - short_spread) * min(t/5, 1)  # 5y ramp
    
    # Integrate over time
    time_points = np.linspace(0, 5, 20)
    spreads = [spread_curve(t) for t in time_points]
    discount_factors = np.exp(-0.03 * time_points)  # Using 3% discount rate
    
    # FVA calculation
    fva_cost = np.sum(epe * spreads * discount_factors * np.gradient(time_points))
    fva_benefit = np.sum(ene * spreads * discount_factors * np.gradient(time_points) * (1 - 0.4))  # 40% recovery
    
    return fva_cost, fva_benefit, time_points, spreads

# Callbacks
@app.callback(
    Output('discount-model-output', 'children'),
    Input('calc-discount-fva', 'n_clicks'),
    [State('libor-rate', 'value'),
     State('funding-spread', 'value'),
     State('notional', 'value'),
     State('maturity', 'value'),
     State('collateral-type', 'value')]
)
def update_discount_model(n_clicks, libor_rate, funding_spread, notional, maturity, collateral_type):
    if n_clicks is None:
        return ""
    
    pv_risk_free, pv_adjusted, fva = calculate_discount_fva(
        libor_rate, funding_spread, notional, maturity, collateral_type
    )
    
    # Create valuation comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Risk-Free', 'Adjusted'],
        y=[pv_risk_free, pv_adjusted],
        name='Present Value'
    ))
    
    fig.update_layout(
        title='First-Generation FVA: Discount Adjustment',
        yaxis_title='Present Value ($M)',
        hovermode='x'
    )
    
    # Create discount curve comparison
    times = np.linspace(0, maturity, 20)
    risk_free_dfs = np.exp(-libor_rate/100 * times)
    
    if collateral_type == 'full':
        adjusted_dfs = risk_free_dfs  # OIS discounting
    elif collateral_type == 'partial':
        adjusted_dfs = np.exp(-(libor_rate/100 + funding_spread/100 * 0.5) * times)
    else:
        adjusted_dfs = np.exp(-(libor_rate/100 + funding_spread/100) * times)
    
    curve_fig = go.Figure()
    curve_fig.add_trace(go.Scatter(
        x=times, y=risk_free_dfs, name='Risk-Free Discount'
    ))
    curve_fig.add_trace(go.Scatter(
        x=times, y=adjusted_dfs, name='Adjusted Discount'
    ))
    
    curve_fig.update_layout(
        title='Discount Curve Comparison',
        xaxis_title='Time (years)',
        yaxis_title='Discount Factor',
        hovermode='x unified'
    )
    
    notes = html.Div([
        html.H5("First-Generation FVA Interpretation"),
        html.P("This shows the simple discount adjustment approach:"),
        html.Ul([
            html.Li(f"Risk-free PV: ${pv_risk_free:.2f}M (discounted at LIBOR)"),
            html.Li(f"Adjusted PV: ${pv_adjusted:.2f}M (includes funding spread)"),
            html.Li(f"FVA Impact: ${fva:.2f}M")
        ]),
        html.P("Key limitations:"),
        html.Ul([
            html.Li("Cannot handle partial collateralization well"),
            html.Li("Leads to DVA double-counting issues"),
            html.Li("No connection to actual exposures")
        ])
    ])
    
    return [
        dcc.Graph(figure=fig),
        dcc.Graph(figure=curve_fig),
        notes
    ]

@app.callback(
    Output('double-counting-output', 'children'),
    Input('analyze-double-counting', 'n_clicks'),
    [State('default-intensity', 'value'),
     State('recovery-rate', 'value'),
     State('bond-cds-basis', 'value'),
     State('counterparty-type', 'value')]
)
def update_double_counting(n_clicks, default_intensity, recovery_rate, bond_cds_basis, counterparty_type):
    if n_clicks is None:
        return ""
    
    cva, fva_adj, dva = analyze_double_counting(
        default_intensity, recovery_rate, bond_cds_basis, counterparty_type
    )
    
    # Create comparison plot
    fig = go.Figure()
    
    if counterparty_type == 'risky_borrower':
        fig.add_trace(go.Bar(
            x=['Lender CVA', 'Borrower FVA Adj', 'Traditional DVA'],
            y=[cva, fva_adj, dva],
            name='Adjustments'
        ))
        title = "Double Counting: Risky Borrower Case"
    else:
        fig.add_trace(go.Bar(
            x=['Lender FVA Adj'],
            y=[fva_adj],
            name='Adjustments'
        ))
        title = "Pure FVA Case: Risky Lender"
    
    fig.update_layout(
        title=title,
        yaxis_title='Adjustment Value',
        hovermode='x'
    )
    
    # Create theoretical comparison
    lambda_b = default_intensity / 100
    times = np.linspace(0, 5, 20)
    cva_curve = 1 - np.exp(-lambda_b * times)
    fva_curve = 1 - np.exp(-(2*lambda_b + bond_cds_basis/100) * times)
    
    theory_fig = go.Figure()
    theory_fig.add_trace(go.Scatter(
        x=times, y=cva_curve, name='CVA/DVA (λ)'
    ))
    theory_fig.add_trace(go.Scatter(
        x=times, y=fva_curve, name='FVA Adjusted (2λ+γ)'
    ))
    
    theory_fig.update_layout(
        title='Theoretical Adjustment Curves',
        xaxis_title='Time (years)',
        yaxis_title='Adjustment',
        hovermode='x unified'
    )
    
    notes = html.Div([
        html.H5("DVA Double Counting Interpretation"),
        html.P("Key findings from equations 9.2-9.10:"),
        html.Ul([
            html.Li("First-gen FVA leads to 2λ term in adjustment (eq 9.7)"),
            html.Li("Traditional DVA only has λ term (eq 9.8)"),
            html.Li("Results in valuation asymmetry between parties")
        ]),
        html.P("Solutions:"),
        html.Ul([
            html.Li("Use exposure-based FVA models (second generation)"),
            html.Li("Explicitly separate funding and credit components"),
            html.Li("Consider bond-CDS basis (γ) separately")
        ])
    ])
    
    return [
        dcc.Graph(figure=fig),
        dcc.Graph(figure=theory_fig),
        notes
    ]

@app.callback(
    Output('exposure-model-output', 'children'),
    Input('calc-exposure-fva', 'n_clicks'),
    [State('epe', 'value'),
     State('ene', 'value'),
     State('short-term-spread', 'value'),
     State('long-term-spread', 'value')]
)
def update_exposure_model(n_clicks, epe, ene, short_term_spread, long_term_spread):
    if n_clicks is None:
        return ""
    
    fva_cost, fva_benefit, time_points, spreads = calculate_exposure_fva(
        epe, ene, short_term_spread, long_term_spread
    )
    
    # Create FVA breakdown
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['FVA Cost', 'FVA Benefit', 'Net FVA'],
        y=[fva_cost, fva_benefit, fva_cost - fva_benefit],
        name='FVA Components'
    ))
    
    fig.update_layout(
        title='Second-Generation FVA Calculation',
        yaxis_title='FVA ($M)',
        hovermode='x'
    )
    
    # Create spread term structure
    spread_fig = go.Figure()
    spread_fig.add_trace(go.Scatter(
        x=time_points, y=np.array(spreads)*100, name='Funding Spread'
    ))
    
    spread_fig.update_layout(
        title='Funding Spread Term Structure',
        xaxis_title='Time (years)',
        yaxis_title='Spread (bps)',
        hovermode='x unified'
    )
    
    # Exposure profile
    exposure_fig = go.Figure()
    exposure_fig.add_trace(go.Scatter(
        x=time_points, y=[epe] * len(time_points), name='EPE', fill='tozeroy'
    ))
    exposure_fig.add_trace(go.Scatter(
        x=time_points, y=[ene] * len(time_points), name='ENE', fill='tozeroy'
    ))
    
    exposure_fig.update_layout(
        title='Exposure Profiles',
        xaxis_title='Time (years)',
        yaxis_title='Exposure ($M)',
        hovermode='x unified'
    )
    
    notes = html.Div([
        html.H5("Second-Generation FVA Interpretation"),
        html.P("Exposure-based FVA models:"),
        html.Ul([
            html.Li(f"FVA Cost: ${fva_cost:.2f}M (funding EPE)"),
            html.Li(f"FVA Benefit: ${fva_benefit:.2f}M (funding ENE)"),
            html.Li(f"Net FVA: ${fva_cost - fva_benefit:.2f}M")
        ]),
        html.P("Advantages over first-gen models:"),
        html.Ul([
            html.Li("Properly handles collateral thresholds"),
            html.Li("No DVA double-counting"),
            html.Li("Links to actual exposures and CSA terms"),
            html.Li("Can incorporate term structure of funding spreads")
        ])
    ])
    
    return [
        dcc.Graph(figure=fig),
        dcc.Graph(figure=spread_fig),
        dcc.Graph(figure=exposure_fig),
        notes
    ]

if __name__ == '__main__':
    app.run(debug=True)