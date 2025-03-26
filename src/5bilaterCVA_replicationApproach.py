import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import integrate

# Initial data
data = {
    'Tenor': np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
    'EPE': np.array([0, 701997, 877137, 914128, 880804, 816664, 739458, 603905, 473606, 290466, 93556]),
    'ENE': np.array([0, -790189, -1138387, -1358724, -1485909, -1542157, -1533924, -1378604, -1146750, -768355, -286472])
}
df = pd.DataFrame(data)

# Initialize Dash app
app = dash.Dash(__name__)

# Calculation functions
def calculate_cva_dva(r, lambda_C, lambda_B, RC, RB):
    def discount_factor(t):
        return np.exp(-integrate.quad(lambda u: r + lambda_C + lambda_B, 0, t)[0])

    def cva_integral(t):
        idx = np.searchsorted(df['Tenor'], t)
        if idx >= len(df['Tenor']):
            return 0
        epe = df['EPE'][idx]
        return -(1 - RC) * lambda_C * discount_factor(t) * max(epe, 0)

    def dva_integral(t):
        idx = np.searchsorted(df['Tenor'], t)
        if idx >= len(df['Tenor']):
            return 0
        ene = df['ENE'][idx]
        return -(1 - RB) * lambda_B * discount_factor(t) * max(-ene, 0)

    tenors = df['Tenor']
    cva_contributions = [0]
    dva_contributions = [0]

    for i in range(1, len(tenors)):
        cva_contrib = integrate.quad(cva_integral, tenors[i-1], tenors[i])[0]
        dva_contrib = integrate.quad(dva_integral, tenors[i-1], tenors[i])[0]
        cva_contributions.append(cva_contrib)
        dva_contributions.append(dva_contrib)

    return cva_contributions, dva_contributions

# App layout
app.layout = html.Div([
    html.H1('Bilateral CVA Visualization - Replication Approach', style={'text-align': 'center'}),
    
    html.Div([
        # Left sidebar
        html.Div([
            html.H3('Input Parameters'),
            
            html.Div([
                html.Label('Risk-free Rate (%):'),
                dcc.Input(id='r-input', type='number', value=2.0, min=0, max=100, step=0.1),
            ], style={'margin-bottom': '15px'}),
            
            html.Div([
                html.Label('Counterparty Hazard Rate (%):'),
                dcc.Input(id='lambda-c-input', type='number', value=3.0, min=0, max=100, step=0.1),
            ], style={'margin-bottom': '15px'}),
            
            html.Div([
                html.Label('Bank Hazard Rate (%):'),
                dcc.Input(id='lambda-b-input', type='number', value=1.0, min=0, max=100, step=0.1),
            ], style={'margin-bottom': '15px'}),
            
            html.Div([
                html.Label('Counterparty Recovery Rate (%):'),
                dcc.Input(id='rc-input', type='number', value=40.0, min=0, max=100, step=1),
            ], style={'margin-bottom': '15px'}),
            
            html.Div([
                html.Label('Bank Recovery Rate (%):'),
                dcc.Input(id='rb-input', type='number', value=40.0, min=0, max=100, step=1),
            ], style={'margin-bottom': '15px'}),
            
            html.Button('Calculate', id='calculate-button', n_clicks=0, 
                       style={'width': '100%', 'padding': '10px', 'margin-bottom': '20px'}),
            
            html.Hr(),
            html.H3('Notes'),
            html.P('This application implements the bilateral CVA replication approach based on equations 3.70-3.71.'),
            html.P('CVA = -(1-RC) ∫ λ_C(t) DF(t) EPE(t) dt'),
            html.P('DVA = -(1-RB) ∫ λ_B(t) DF(t) ENE(t) dt'),
            html.P('Where DF(t) = exp(-∫(r + λ_C + λ_B)du)'),
            html.P('Values are calculated using numerical integration over discrete tenor points.')
        ], style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '20px'}),
        
        # Main content
        html.Div([
            html.Div(id='results-output'),
            dcc.Graph(id='exposure-graph'),
            dcc.Graph(id='cva-dva-graph'),
        ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'})
    ])
])

# Callback to update graphs and results when button is clicked
@app.callback(
    [Output('exposure-graph', 'figure'),
     Output('cva-dva-graph', 'figure'),
     Output('results-output', 'children')],
    [Input('calculate-button', 'n_clicks')],
    [State('r-input', 'value'),
     State('lambda-c-input', 'value'),
     State('lambda-b-input', 'value'),
     State('rc-input', 'value'),
     State('rb-input', 'value')]
)
def update_graphs(n_clicks, r, lambda_c, lambda_b, rc, rb):
    # Default empty figures if not calculated yet
    fig1 = go.Figure()
    fig1.update_layout(
        title='Expected Exposure Profiles',
        xaxis_title='Tenor (years)',
        yaxis_title='Exposure',
        template='plotly_white'
    )
    
    fig2 = go.Figure()
    fig2.update_layout(
        title='CVA and DVA Contributions',
        xaxis_title='Tenor (years)',
        yaxis_title='Contribution',
        template='plotly_white'
    )
    
    results = html.Div("Press Calculate to see results", style={'text-align': 'center'})

    if n_clicks > 0:
        # Convert percentages to decimals
        r = r / 100 if r is not None else 0.02
        lambda_c = lambda_c / 100 if lambda_c is not None else 0.03
        lambda_b = lambda_b / 100 if lambda_b is not None else 0.01
        rc = rc / 100 if rc is not None else 0.4
        rb = rb / 100 if rb is not None else 0.4

        # Calculate CVA and DVA
        cva_contribs, dva_contribs = calculate_cva_dva(r, lambda_c, lambda_b, rc, rb)
        df['CVA_Contribution'] = cva_contribs
        df['DVA_Contribution'] = dva_contribs
        
        total_cva = sum(cva_contribs)
        total_dva = sum(dva_contribs)
        bcva = total_cva + total_dva

        # Exposure graph
        fig1.add_trace(go.Scatter(x=df['Tenor'], y=df['EPE'], name='EPE', mode='lines+markers'))
        fig1.add_trace(go.Scatter(x=df['Tenor'], y=df['ENE'], name='ENE', mode='lines+markers'))

        # CVA/DVA graph
        fig2.add_trace(go.Bar(x=df['Tenor'], y=df['CVA_Contribution'], name='CVA Contribution'))
        fig2.add_trace(go.Bar(x=df['Tenor'], y=df['DVA_Contribution'], name='DVA Contribution'))
        fig2.update_layout(barmode='relative')

        # Results
        results = html.Div([
            html.H3(f'Total CVA: {total_cva:,.2f}'),
            html.H3(f'Total DVA: {total_dva:,.2f}'),
            html.H3(f'BCVA: {bcva:,.2f}'),
        ], style={'text-align': 'center'})

    return fig1, fig2, results

# Run the app
if __name__ == '__main__':
    app.run(debug=True)