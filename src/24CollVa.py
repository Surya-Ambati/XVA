import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from collections import defaultdict

# Dummy data generation
def generate_dummy_data():
    dates = pd.date_range(start='2025-01-01', periods=5, freq='ME')
    
    # Market data
    market_data = {
        'collva_spread_curves': {
            1: [(0.5, 0.01), (1.0, 0.015), (2.0, 0.02), (3.0, 0.025), (5.0, 0.03)],
            2: [(0.5, 0.012), (1.0, 0.017), (2.0, 0.022), (3.0, 0.027), (5.0, 0.032)]
        },
        'xccy_spread_curves': {
            0: [(0.5, 0.005), (1.0, 0.007), (2.0, 0.01), (3.0, 0.015), (5.0, 0.02)]
        }
    }
    
    # Simulation results
    sim_results = {
        'NET1': {
            'bank_grid_id': '12345',
            'cty_grid_id': '1',
            'exposure_dates': dates,
            'profiles': {
                'EPE_DISC_TPLUS1': [100, 120, 130, 140, 150],
                'ENE_DISC_TPLUS1': [-80, -90, -100, -110, -120],
                'EPC': [90, 100, 110, 120, 130],
                'ENC': [-70, -80, -90, -100, -110]
            }
        },
        'NET2': {
            'bank_grid_id': '67890',
            'cty_grid_id': '2',
            'exposure_dates': dates,
            'profiles': {
                'EPE_DISC_TPLUS1': [80, 90, 100, 110, 120],
                'ENE_DISC_TPLUS1': [-60, -70, -80, -90, -100],
                'EPC': [70, 80, 90, 100, 110],
                'ENC': [-50, -60, -70, -80, -90]
            }
        }
    }
    
    netting_pool_mapping = {
        'NET1': {'remun_collva': True, 'xccy_collva': True},
        'NET2': {'remun_collva': True, 'xccy_collva': True}
    }
    
    return market_data, sim_results, netting_pool_mapping

# CollVA Calculation function
def calculate_collva(market_data, sim_results, netting_pool_mapping):
    """
    Parameters:
    - market_data: dict containing collva_spread_curves and xccy_spread_curves
        * collva_spread_curves: Mapping of counterparty IDs to spread curves (tenor, rate)
        * xccy_spread_curves: Cross-currency spread curves
    - sim_results: dict containing simulation data per netting set
        * bank_grid_id: Bank grid identifier
        * cty_grid_id: Counterparty grid identifier
        * exposure_dates: List of dates for exposure profiles
        * profiles: Exposure profiles (EPE, ENE, EPC, ENC)
    - netting_pool_mapping: dict indicating which calculations to perform per netting set
        * remun_collva: Boolean for remuneration CollVA
        * xccy_collva: Boolean for cross-currency CollVA
    
    Returns:
    - results: dict containing calculated CollVA benefits and costs
    """
    results = {}
    
    for net_set_id, sim_data in sim_results.items():
        collva_rem_benefit = 0.0
        collva_rem_cost = 0.0
        collva_xccy_benefit = 0.0
        collva_xccy_cost = 0.0
        
        # Get curve data
        cty_id = int(sim_data['cty_grid_id'])
        collva_curve = sorted(market_data['collva_spread_curves'].get(cty_id, []))
        xccy_curve = sorted(market_data['xccy_spread_curves'][0])
        
        # Check netting pool scope
        mapping = netting_pool_mapping[net_set_id]
        calc_rem = mapping['remun_collva'] and collva_curve
        calc_xccy = mapping['xccy_collva']
        
        # Get profiles
        profiles = sim_data['profiles']
        nb_dates = len(sim_data['exposure_dates'])
        
        for i in range(nb_dates):
            epe_disc = profiles['EPE_DISC_TPLUS1'][i]
            ene_disc = profiles['ENE_DISC_TPLUS1'][i]
            epc = profiles['EPC'][i]
            enc = profiles['ENC'][i]
            
            # CollVA Remuneration calculations
            if calc_rem:
                collva_rem_benefit -= epc * collva_curve[i][1]
                collva_rem_cost -= enc * collva_curve[i][1]
            
            # CollVA Cross-currency calculations
            if calc_xccy:
                idx = min(i + 1, nb_dates - 1)
                if sim_data['bank_grid_id'] == '12345':
                    collva_xccy_benefit += epe_disc * xccy_curve[idx][1]
                    collva_xccy_cost += ene_disc * xccy_curve[idx][1]
                elif sim_data['bank_grid_id'] == '67890':
                    collva_xccy_benefit -= epc * xccy_curve[idx][1]
                    collva_xccy_cost -= enc * xccy_curve[idx][1]
        
        results[net_set_id] = {
            'collva_rem_benefit': collva_rem_benefit,
            'collva_rem_cost': collva_rem_cost,
            'collva_xccy_benefit': collva_xccy_benefit,
            'collva_xccy_cost': collva_xccy_cost
        }
    
    return results

# Dash application
app = dash.Dash(__name__)

app.layout = html.Div([
    # Left panel (Inputs)
    html.Div([
        html.H3("CollVA Calculator"),
        dcc.Dropdown(
            id='netting-set-dropdown',
            options=[{'label': k, 'value': k} for k in ['NET1', 'NET2']],
            value='NET1'
        ),
        html.Button('Calculate', id='calculate-btn', n_clicks=0),
        html.Hr(),
        html.H4("Input Parameters"),
        html.P("Select netting set and click Calculate to see results")
    ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '20px'}),
    
    # Right panel (Results and Graphs)
    html.Div([
        html.H3("Results"),
        html.Div(id='results-output'),
        dcc.Graph(id='benefit-cost-graph'),
        dcc.Graph(id='exposure-profile-graph'),
        html.H4("Notes and Interpretation"),
        dcc.Markdown("""
            ### Graph Interpretation:
            1. **Benefit vs Cost Graph**: 
               - Shows comparison between benefits and costs for both remuneration and cross-currency CollVA
               - Positive values indicate benefits, negative values indicate costs
            
            2. **Exposure Profile Graph**:
               - Displays EPE, ENE, EPC, and ENC profiles over time
               - Helps understand the exposure patterns driving CollVA calculations
            
            ### Notes:
            - CollVA Remuneration reflects the collateral value adjustment based on counterparty curves
            - Cross-currency CollVA varies based on Bank grid ID (12345 or 67890)
            - Results are sensitive to spread curve values and exposure profiles
        """)
    ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px'})
])

@app.callback(
    [Output('results-output', 'children'),
     Output('benefit-cost-graph', 'figure'),
     Output('exposure-profile-graph', 'figure')],
    [Input('calculate-btn', 'n_clicks')],
    [State('netting-set-dropdown', 'value')]
)
def update_output(n_clicks, selected_net_set):
    if n_clicks == 0:
        return "Click Calculate to see results", go.Figure(), go.Figure()
    
    # Generate dummy data and calculate
    market_data, sim_results, netting_pool_mapping = generate_dummy_data()
    results = calculate_collva(market_data, sim_results, netting_pool_mapping)
    
    # Results for selected netting set
    result = results[selected_net_set]
    results_text = [
        html.P(f"CollVA Rem Benefit: {result['collva_rem_benefit']:.2f}"),
        html.P(f"CollVA Rem Cost: {result['collva_rem_cost']:.2f}"),
        html.P(f"CollVA XCcy Benefit: {result['collva_xccy_benefit']:.2f}"),
        html.P(f"CollVA XCcy Cost: {result['collva_xccy_cost']:.2f}")
    ]
    
    # Benefit vs Cost Graph
    benefit_cost_fig = go.Figure(data=[
        go.Bar(name='Rem Benefit', x=['CollVA'], y=[result['collva_rem_benefit']]),
        go.Bar(name='Rem Cost', x=['CollVA'], y=[result['collva_rem_cost']]),
        go.Bar(name='XCcy Benefit', x=['CollVA'], y=[result['collva_xccy_benefit']]),
        go.Bar(name='XCcy Cost', x=['CollVA'], y=[result['collva_xccy_cost']])
    ])
    benefit_cost_fig.update_layout(barmode='group', title='Benefits vs Costs')
    
    # Exposure Profile Graph
    dates = sim_results[selected_net_set]['exposure_dates']
    profiles = sim_results[selected_net_set]['profiles']
    exposure_fig = go.Figure()
    for profile_name, values in profiles.items():
        exposure_fig.add_trace(go.Scatter(x=dates, y=values, name=profile_name))
    exposure_fig.update_layout(title='Exposure Profiles Over Time')
    
    return results_text, benefit_cost_fig, exposure_fig

if __name__ == '__main__':
    app.run(debug=True)