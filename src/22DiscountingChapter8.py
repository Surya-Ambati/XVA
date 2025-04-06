import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import curve_fit

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            # Control Panel
            html.H3("XVA Discount Curve Controls"),
            dcc.Tabs([
                # Single Curve Tab
                dcc.Tab(label="Single Curve", children=[
                    html.H5("Deposit Rates"),
                    dcc.Input(id='deposit-6m', value=0.02, type='number'),
                    dcc.Input(id='deposit-1y', value=0.022, type='number'),
                    
                    html.H5("Futures Rates"),
                    dcc.Input(id='future-6m', value=0.021, type='number'),
                    dcc.Input(id='future-1y', value=0.023, type='number'),
                    
                    html.H5("Swap Rates"),
                    dcc.Input(id='swap-2y', value=0.025, type='number'),
                    dcc.Input(id='swap-5y', value=0.028, type='number'),
                    
                    html.Button('Build Curve', id='build-single-curve'),
                ]),
                
                # Multi-Curve Tab
                dcc.Tab(label="Multi-Curve", children=[
                    html.H5("OIS Curve"),
                    dcc.Input(id='ois-1y', value=0.01, type='number'),
                    dcc.Input(id='ois-5y', value=0.015, type='number'),
                    
                    html.H5("3M LIBOR Curve"),
                    dcc.Input(id='libor3m-1y', value=0.02, type='number'),
                    dcc.Input(id='libor3m-5y', value=0.025, type='number'),
                    
                    html.H5("6M LIBOR Curve"),
                    dcc.Input(id='libor6m-1y', value=0.021, type='number'),
                    dcc.Input(id='libor6m-5y', value=0.026, type='number'),
                    
                    html.H5("Basis Spread (3M-6M)"),
                    dcc.Input(id='basis-1y', value=0.005, type='number'),
                    
                    html.Button('Build Multi-Curve', id='build-multi-curve'),
                ]),
                
                # CSA Discounting Tab
                dcc.Tab(label="CSA Discounting", children=[
                    html.H5("OIS Rates"),
                    dcc.Input(id='ois-csa-1y', value=0.01, type='number'),
                    dcc.Input(id='ois-csa-5y', value=0.015, type='number'),
                    
                    html.H5("Funding Spread"),
                    dcc.Input(id='funding-spread', value=0.02, type='number'),
                    
                    html.Button('Calculate CSA Value', id='calculate-csa'),
                ]),
                
                # Collateral Option Tab
                dcc.Tab(label="Collateral Option", children=[
                    html.H5("Domestic Collateral Rate"),
                    dcc.Input(id='domestic-rate', value=-0.005, type='number'),
                    
                    html.H5("Foreign Collateral Rate"),
                    dcc.Input(id='foreign-rate', value=0.015, type='number'),
                    
                    html.H5("Cross-Currency Basis"),
                    dcc.Input(id='ccy-basis', value=0.008, type='number'),
                    
                    html.Button('Calculate CTD', id='calculate-ctd'),
                ])
            ]),
        ], width=4),
        
        # Results Panel
        dbc.Col([
            html.Div(id='single-curve-output'),
            html.Div(id='multi-curve-output'),
            html.Div(id='csa-output'),
            html.Div(id='ctd-output'),
        ], width=8)
    ])
])

# Helper functions
def bootstrap_curve(tenors, rates, day_count=1.0):
    """Bootstrap a discount curve using log-linear interpolation"""
    dfs = [1 / (1 + rates[0] * tenors[0])]
    for i in range(1, len(tenors)):
        df = dfs[-1] / (1 + rates[i] * (tenors[i] - tenors[i-1]))
        dfs.append(df)
    return np.array(dfs)

def calculate_ctd(domestic_rate, foreign_rate, ccy_basis):
    effective_rate = max(domestic_rate, foreign_rate + ccy_basis)
    option_value = max(foreign_rate + ccy_basis - domestic_rate, 0)
    return effective_rate, option_value

# Callbacks
@app.callback(
    Output('single-curve-output', 'children'),
    Input('build-single-curve', 'n_clicks'),
    [State('deposit-6m', 'value'),
     State('deposit-1y', 'value'),
     State('future-6m', 'value'),
     State('future-1y', 'value'),
     State('swap-2y', 'value'),
     State('swap-5y', 'value')]
)
def build_single_curve(n_clicks, deposit_6m, deposit_1y, future_6m, future_1y, swap_2y, swap_5y):
    if n_clicks is None:
        return ""
    
    # Create sample data
    tenors = np.array([0.5, 1.0, 1.5, 2.0, 5.0])
    rates = np.array([deposit_6m, deposit_1y, future_6m, future_1y, swap_2y, swap_5y])
    
    # Bootstrap curve
    dfs = bootstrap_curve(tenors, rates)
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tenors, y=dfs, name='Discount Factors', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=tenors, y=rates, name='Input Rates', yaxis='y2', mode='markers'))
    
    fig.update_layout(
        title='Single Curve Construction',
        yaxis=dict(title='Discount Factor'),
        yaxis2=dict(title='Rate', overlaying='y', side='right'),
        xaxis=dict(title='Tenor (years)'),
        hovermode='x unified'
    )
    
    notes = html.Div([
        html.H5("Single Curve Interpretation"),
        html.P("This shows the traditional single-curve approach where all rates (deposits, futures, swaps) are used to build one discount curve."),
        html.P("Key points:"),
        html.Ul([
            html.Li("Deposits define the short end of the curve"),
            html.Li("Futures (with convexity adjustment) extend the curve"),
            html.Li("Swaps complete the long end via bootstrap"),
            html.Li("Log-linear interpolation ensures positive forward rates")
        ])
    ])
    
    return [dcc.Graph(figure=fig), notes]

@app.callback(
    Output('multi-curve-output', 'children'),
    Input('build-multi-curve', 'n_clicks'),
    [State('ois-1y', 'value'),
     State('ois-5y', 'value'),
     State('libor3m-1y', 'value'),
     State('libor3m-5y', 'value'),
     State('libor6m-1y', 'value'),
     State('libor6m-5y', 'value'),
     State('basis-1y', 'value')]
)
def build_multi_curve(n_clicks, ois_1y, ois_5y, libor3m_1y, libor3m_5y, libor6m_1y, libor6m_5y, basis_1y):
    if n_clicks is None:
        return ""
    
    # Create sample data
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
    
    # OIS curve (discounting)
    ois_rates = np.array([0.01, ois_1y, ois_5y])
    ois_dfs = bootstrap_curve(np.array([0.5, 1.0, 5.0]), ois_rates)
    
    # 3M LIBOR curve
    libor3m_rates = np.array([libor3m_1y, libor3m_5y])
    libor3m_dfs = bootstrap_curve(np.array([1.0, 5.0]), libor3m_rates)
    
    # 6M LIBOR curve (including basis)
    libor6m_rates = np.array([libor6m_1y, libor6m_5y])
    libor6m_dfs = bootstrap_curve(np.array([1.0, 5.0]), libor6m_rates)
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.5, 1.0, 5.0], y=ois_dfs, name='OIS Discount Curve', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=[1.0, 5.0], y=libor3m_dfs, name='3M LIBOR Curve', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=[1.0, 5.0], y=libor6m_dfs, name='6M LIBOR Curve', mode='lines+markers'))
    
    fig.update_layout(
        title='Multi-Curve Framework',
        yaxis=dict(title='Discount Factor'),
        xaxis=dict(title='Tenor (years)'),
        hovermode='x unified'
    )
    
    # Basis plot
    basis_fig = go.Figure()
    basis_fig.add_trace(go.Scatter(x=[1.0], y=[basis_1y], name='3M-6M Basis', mode='markers'))
    
    basis_fig.update_layout(
        title='Tenor Basis Spread',
        yaxis=dict(title='Basis (bps)'),
        xaxis=dict(title='Tenor (years)')
    )
    
    notes = html.Div([
        html.H5("Multi-Curve Interpretation"),
        html.P("Post-crisis, we use separate curves for discounting (OIS) and forwarding (LIBOR)."),
        html.P("Key points:"),
        html.Ul([
            html.Li("OIS curve used for discounting (risk-free rate proxy)"),
            html.Li("Separate forwarding curves for each tenor (3M, 6M LIBOR)"),
            html.Li("Tenor basis reflects liquidity/credit differences"),
            html.Li("Basis swaps used to calibrate forwarding curves")
        ])
    ])
    
    return [dcc.Graph(figure=fig), dcc.Graph(figure=basis_fig), notes]

@app.callback(
    Output('csa-output', 'children'),
    Input('calculate-csa', 'n_clicks'),
    [State('ois-csa-1y', 'value'),
     State('ois-csa-5y', 'value'),
     State('funding-spread', 'value')]
)
def calculate_csa_value(n_clicks, ois_1y, ois_5y, funding_spread):
    if n_clicks is None:
        return ""
    
    # Create sample data
    tenors = np.array([0.5, 1.0, 5.0])
    ois_rates = np.array([0.01, ois_1y, ois_5y])
    ois_dfs = bootstrap_curve(tenors, ois_rates)
    funding_rates = ois_rates + funding_spread
    
    # Calculate CSA vs non-CSA values
    swap_csa = 100 * ois_dfs[-1]  # Simple example
    swap_non_csa = 100 * (ois_dfs[-1] * np.exp(-funding_spread * tenors[-1]))
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tenors, y=ois_rates, name='OIS Rates', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=tenors, y=funding_rates, name='Funding Rates', mode='lines+markers'))
    
    fig.update_layout(
        title='CSA Discounting vs Funding Rates',
        yaxis=dict(title='Rate'),
        xaxis=dict(title='Tenor (years)'),
        hovermode='x unified'
    )
    
    notes = html.Div([
        html.H5("CSA Discounting Interpretation"),
        html.P("For CSA trades, we discount at the collateral rate (OIS). For non-CSA trades, we use the bank's funding rate."),
        html.P(f"Example 5Y swap valuation:"),
        html.Ul([
            html.Li(f"CSA value: {swap_csa:.2f} (discounted at OIS)"),
            html.Li(f"Non-CSA value: {swap_non_csa:.2f} (includes FVA)"),
            html.Li(f"FVA impact: {swap_non_csa-swap_csa:.2f}")
        ]),
        html.P("Key points:"),
        html.Ul([
            html.Li("Full collateralization → discount at collateral rate"),
            html.Li("Unsecured trades → discount at funding rate"),
            html.Li("Funding spread reflects bank's credit risk")
        ])
    ])
    
    return [dcc.Graph(figure=fig), notes]

@app.callback(
    Output('ctd-output', 'children'),
    Input('calculate-ctd', 'n_clicks'),
    [State('domestic-rate', 'value'),
     State('foreign-rate', 'value'),
     State('ccy-basis', 'value')]
)
def calculate_ctd_value(n_clicks, domestic_rate, foreign_rate, ccy_basis):
    if n_clicks is None:
        return ""
    
    effective_rate, option_value = calculate_ctd(domestic_rate, foreign_rate, ccy_basis)
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Domestic', 'Foreign+CCY', 'Effective'],
        y=[domestic_rate, foreign_rate+ccy_basis, effective_rate],
        name='Rates'
    ))
    
    fig.update_layout(
        title='Cheapest-to-Deliver Collateral Calculation',
        yaxis=dict(title='Rate'),
        hovermode='x unified'
    )
    
    notes = html.Div([
        html.H5("Collateral Option Interpretation"),
        html.P("When multiple collateral currencies are allowed, the posting party will choose the cheapest option."),
        html.P(f"Calculation:"),
        html.Ul([
            html.Li(f"Domestic collateral rate: {domestic_rate:.2%}"),
            html.Li(f"Foreign collateral rate + basis: {foreign_rate+ccy_basis:.2%}"),
            html.Li(f"Effective rate: {effective_rate:.2%}"),
            html.Li(f"Option value: {option_value:.2%}")
        ]),
        html.P("Key points:"),
        html.Ul([
            html.Li("Cross-currency basis adjusts for FX risk"),
            html.Li("Negative rates make CTD calculation non-trivial"),
            html.Li("Option becomes valuable when basis is large")
        ])
    ])
    
    return [dcc.Graph(figure=fig), notes]

if __name__ == '__main__':
    app.run(debug=True)