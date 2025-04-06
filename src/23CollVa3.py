import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Helper functions
def create_dummy_data():
    """Create dummy data for demonstration purposes"""
    # Create dates for exposure profiles
    dates = [datetime.today() + timedelta(days=30*i) for i in range(12)]
    
    # Create dummy curves
    tenors = [i/12 for i in range(1, 13)]  # 1 month to 12 months
    collVA_curve = {t: 0.01 * np.exp(-0.1*t) for t in tenors}
    xCcy_curve = {t: 0.015 * np.exp(-0.08*t) for t in tenors}
    
    # Create dummy exposure profiles
    num_results = 5
    profiles = {
        'EPE_DISC_TPLUS1': [np.random.uniform(0, 1e6) for _ in range(len(dates))],
        'ENE_DISC_TPLUS1': [np.random.uniform(-1e6, 0) for _ in range(len(dates))],
        'EPC': [np.random.uniform(0, 5e5) for _ in range(len(dates))],
        'ENC': [np.random.uniform(-5e5, 0) for _ in range(len(dates))]
    }
    
    # Create netting pool mapping data
    netting_pool_mapping = {
        ('Portfolio1', 'Pool1'): {'RemunCollVA': True, 'XccyCollVA': True},
        ('Portfolio2', 'Pool2'): {'RemunCollVA': False, 'XccyCollVA': True}
    }
    
    return {
        'dates': dates,
        'collVA_curve': collVA_curve,
        'xCcy_curve': xCcy_curve,
        'profiles': profiles,
        'netting_pool_mapping': netting_pool_mapping,
        'num_results': num_results
    }

def calculate_collva(data, bank_grid_id, cty_grid_id, net_set_id, incremental_group_index):
    """
    Calculate CollVA metrics based on input data
    
    Parameters:
    -----------
    data : dict
        Contains all input data including curves and exposure profiles
    Bank_grid_id : str
        Bank grid identifier (counterparty identifier)
    cty_grid_id : str
        Counterparty grid identifier (used to select the appropriate curve)
    net_set_id : str
        Netting pool identifier (used to check calculation scope)
    incremental_group_index : int
        Index for incremental grouping (used with net_set_id to determine calculation scope)
        
    Returns:
    --------
    dict
        Dictionary containing calculated CollVA metrics
    """
    # Get netting pool mapping information
    portfolio = f"Portfolio{incremental_group_index % 2 + 1}"
    pool = net_set_id
    mapping = data['netting_pool_mapping'].get((portfolio, pool), {'RemunCollVA': False, 'XccyCollVA': False})
    
    # Prepare curves - convert to sorted list of (tenor, value) pairs
    try:
        cty_id = int(cty_grid_id)
        # Create sorted list from the curve dictionary
        collVA_curve = sorted(data['collVA_curve'].items())
        # Apply scaling based on counterparty ID
        collVA_curve = [(t, (1 + cty_id/1000) * v) for t, v in collVA_curve]
    except ValueError:
        collVA_curve = sorted(data['collVA_curve'].items())
    
    xCcy_curve = sorted(data['xCcy_curve'].items())
    
    # Initialize results
    tempCollVARemBenefit = 0.0
    tempCollVARemCost = 0.0
    tempCollVAXCcyBenefit = 0.0
    tempCollVAXCcyCost = 0.0
    
    # Get exposure profiles
    epe_disc = data['profiles']['EPE_DISC_TPLUS1']
    ene_disc = data['profiles']['ENE_DISC_TPLUS1']
    epc = data['profiles']['EPC']
    enc = data['profiles']['ENC']
    
    # Calculate CollVA metrics
    for exp_count in range(len(data['dates'])):
        epe = epe_disc[exp_count]
        ene = ene_disc[exp_count]
        epc_val = epc[exp_count]
        enc_val = enc[exp_count]
        
        exp_count_plus_one = exp_count + 1
        if exp_count_plus_one == len(data['dates']):
            exp_count_plus_one -= 1
            
        if mapping['RemunCollVA']:
            # Now collVA_curve is a list, so we can index by position
            tempCollVARemBenefit -= epc_val * collVA_curve[exp_count][1]
            tempCollVARemCost -= enc_val * collVA_curve[exp_count][1]
        
        if mapping['XccyCollVA']:
            if bank_grid_id == "12345":
                tempCollVAXCcyBenefit += epe * xCcy_curve[exp_count_plus_one][1]
                tempCollVAXCcyCost += ene * xCcy_curve[exp_count_plus_one][1]
            elif bank_grid_id == "67890":
                tempCollVAXCcyBenefit -= epc_val * xCcy_curve[exp_count_plus_one][1]
                tempCollVAXCcyCost -= enc_val * xCcy_curve[exp_count_plus_one][1]
    
    return {
        'COLLVA_REMUN_BENEFIT': tempCollVARemBenefit,
        'COLLVA_REMUN_COST': tempCollVARemCost,
        'COLLVA_XCCY_BENEFIT': tempCollVAXCcyBenefit,
        'COLLVA_XCCY_COST': tempCollVAXCcyCost
    }

# Create dummy data
dummy_data = create_dummy_data()

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("CollVA Calculation Parameters"),
            html.Hr(),
            
            # Input controls
            html.Label("Bank Grid ID:"),
            dcc.Dropdown(
                id='bank-grid-id',
                options=[
                    {'label': '12345', 'value': '12345'},
                    {'label': '67890', 'value': '67890'},
                    {'label': 'Other', 'value': 'other'}
                ],
                value='12345'
            ),
            
            html.Label("Counterparty Grid ID:"),
            dcc.Input(id='cty-grid-id', type='text', value='1'),
            
            html.Label("Netting Pool ID:"),
            dcc.Dropdown(
                id='net-set-id',
                options=[
                    {'label': 'Pool1', 'value': 'Pool1'},
                    {'label': 'Pool2', 'value': 'Pool2'}
                ],
                value='Pool1'
            ),
            
            html.Label("Incremental Group Index:"),
            dcc.Slider(id='incremental-group-index', min=0, max=4, step=1, value=0),
            
            html.Br(),
            dbc.Button("Calculate CollVA", id='calculate-button', color="primary"),
            
            html.Hr(),
            html.H4("Curve Parameters"),
            
            html.Label("CollVA Curve Base Rate (%):"),
            dcc.Slider(id='collva-base-rate', min=0.1, max=2.0, step=0.1, value=1.0),
            
            html.Label("CollVA Curve Decay:"),
            dcc.Slider(id='collva-decay', min=0.01, max=0.2, step=0.01, value=0.1),
            
            html.Label("XCcy Curve Base Rate (%):"),
            dcc.Slider(id='xccy-base-rate', min=0.1, max=2.0, step=0.1, value=1.5),
            
            html.Label("XCcy Curve Decay:"),
            dcc.Slider(id='xccy-decay', min=0.01, max=0.2, step=0.01, value=0.08),
            
        ], width=4),
        
        dbc.Col([
            html.H3("Results"),
            html.Hr(),
            
            # Results display
            html.Div(id='results-display'),
            
            html.Hr(),
            html.H3("Exposure Profiles"),
            dcc.Graph(id='exposure-profiles-graph'),
            
            html.Hr(),
            html.H3("CollVA Curves"),
            dcc.Graph(id='collva-curves-graph'),
            
            html.Hr(),
            html.H3("Notes and Interpretation"),
            dcc.Markdown('''
            ### CollVA Calculation Notes
            
            - **CollVA (Collateral Valuation Adjustment)**: An adjustment to account for the cost or benefit of posting or receiving collateral.
            
            - **Remuneration CollVA**: Adjustment for the interest paid or received on posted collateral.
            
            - **Cross-Currency (XCcy) CollVA**: Adjustment for collateral posted in a different currency than the trade currency.
            
            ### Graph Interpretation
            
            - **Exposure Profiles**: Show the expected positive/negative exposures over time.
              - EPE: Expected Positive Exposure
              - ENE: Expected Negative Exposure
              - EPC: Expected Positive Collateral
              - ENC: Expected Negative Collateral
            
            - **CollVA Curves**: Show the spread curves used for discounting.
              - Higher spreads lead to larger valuation adjustments.
              - The decay rate affects how quickly spreads decrease over time.
            ''')
        ], width=8)
    ])
], fluid=True)

# Callbacks
@app.callback(
    Output('results-display', 'children'),
    Output('exposure-profiles-graph', 'figure'),
    Output('collva-curves-graph', 'figure'),
    Input('calculate-button', 'n_clicks'),
    State('bank-grid-id', 'value'),
    State('cty-grid-id', 'value'),
    State('net-set-id', 'value'),
    State('incremental-group-index', 'value'),
    State('collva-base-rate', 'value'),
    State('collva-decay', 'value'),
    State('xccy-base-rate', 'value'),
    State('xccy-decay', 'value')
)
def update_output(n_clicks, bank_grid_id, cty_grid_id, net_set_id, group_index, 
                 collva_base, collva_decay, xccy_base, xccy_decay):
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Update curves based on user input
    tenors = [i/12 for i in range(1, 13)]
    dummy_data['collVA_curve'] = {t: (collva_base/100) * np.exp(-collva_decay*t) for t in tenors}
    dummy_data['xCcy_curve'] = {t: (xccy_base/100) * np.exp(-xccy_decay*t) for t in tenors}
    
    # Calculate CollVA
    results = calculate_collva(
        dummy_data, bank_grid_id, cty_grid_id, net_set_id, group_index
    )
    
    # Create results display
    results_display = [
        html.H4(f"Results for {bank_grid_id}/{cty_grid_id} in {net_set_id} (Group {group_index})"),
        dbc.Table([
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
            html.Tbody([
                html.Tr([html.Td("Remuneration Benefit"), html.Td(f"${results['COLLVA_REMUN_BENEFIT']:,.2f}")]),
                html.Tr([html.Td("Remuneration Cost"), html.Td(f"${results['COLLVA_REMUN_COST']:,.2f}")]),
                html.Tr([html.Td("XCcy Benefit"), html.Td(f"${results['COLLVA_XCCY_BENEFIT']:,.2f}")]),
                html.Tr([html.Td("XCcy Cost"), html.Td(f"${results['COLLVA_XCCY_COST']:,.2f}")]),
            ])
        ], bordered=True)
    ]
    
    # Create exposure profiles graph
    exposure_fig = go.Figure()
    exposure_fig.add_trace(go.Scatter(
        x=dummy_data['dates'],
        y=dummy_data['profiles']['EPE_DISC_TPLUS1'],
        name='EPE (Discounted)',
        line=dict(color='blue')
    ))
    exposure_fig.add_trace(go.Scatter(
        x=dummy_data['dates'],
        y=dummy_data['profiles']['ENE_DISC_TPLUS1'],
        name='ENE (Discounted)',
        line=dict(color='red')
    ))
    exposure_fig.add_trace(go.Scatter(
        x=dummy_data['dates'],
        y=dummy_data['profiles']['EPC'],
        name='EPC',
        line=dict(color='green')
    ))
    exposure_fig.add_trace(go.Scatter(
        x=dummy_data['dates'],
        y=dummy_data['profiles']['ENC'],
        name='ENC',
        line=dict(color='orange')
    ))
    exposure_fig.update_layout(
        title='Exposure Profiles Over Time',
        xaxis_title='Date',
        yaxis_title='Exposure Amount',
        hovermode='x unified'
    )
    
    # Create curves graph
    curves_fig = go.Figure()
    curves_fig.add_trace(go.Scatter(
        x=list(dummy_data['collVA_curve'].keys()),
        y=list(dummy_data['collVA_curve'].values()),
        name='CollVA Curve'
    ))
    curves_fig.add_trace(go.Scatter(
        x=list(dummy_data['xCcy_curve'].keys()),
        y=list(dummy_data['xCcy_curve'].values()),
        name='XCcy Curve'
    ))
    curves_fig.update_layout(
        title='CollVA Spread Curves',
        xaxis_title='Tenor (Years)',
        yaxis_title='Spread',
        hovermode='x unified'
    )
    
    return results_display, exposure_fig, curves_fig

if __name__ == '__main__':
    app.run(debug=True)