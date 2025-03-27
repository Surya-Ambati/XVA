import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import integrate
from scipy.stats import norm, multivariate_normal

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
def calculate_cva_dva(r, lambda_C, lambda_B, RC, RB, rho):
    def discount_factor(t):
        return np.exp(-integrate.quad(lambda u: r + lambda_C + lambda_B, 0, t)[0])

    def survival_prob(t, lambda_):
        return np.exp(-lambda_ * t)

    def gaussian_copula_survival(t1, t2, lambda_C, lambda_B, rho):
        u = survival_prob(t1, lambda_C)
        v = survival_prob(t2, lambda_B)
        z1 = norm.ppf(1 - u)  # Inverse CDF for counterparty
        z2 = norm.ppf(1 - v)  # Inverse CDF for bank
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        return multivariate_normal.cdf([z1, z2], mean=mean, cov=cov)

    tenors = df['Tenor']
    cva_contributions = [0]
    dva_contributions = [0]

    for i in range(1, len(tenors)):
        t_start = tenors[i-1]
        t_end = tenors[i]
        
        # Midpoint approximation for exposure
        t_mid = (t_start + t_end) / 2
        idx = np.searchsorted(df['Tenor'], t_mid)
        epe = df['EPE'][idx] if idx < len(df['Tenor']) else 0
        ene = df['ENE'][idx] if idx < len(df['Tenor']) else 0

        # Joint survival probabilities using Gaussian Copula
        survival_C = survival_prob(t_end, lambda_C)
        survival_B = survival_prob(t_end, lambda_B)
        joint_survival = gaussian_copula_survival(t_end, t_end, lambda_C, lambda_B, rho)
        
        # Default probabilities over interval
        delta_t = t_end - t_start
        prob_C_default = (survival_prob(t_start, lambda_C) - survival_C) * (survival_B / joint_survival if joint_survival > 0 else 1)
        prob_B_default = (survival_prob(t_start, lambda_B) - survival_B) * (survival_C / joint_survival if joint_survival > 0 else 1)

        # CVA and DVA contributions
        cva_contrib = -(1 - RC) * prob_C_default * discount_factor(t_mid) * max(epe, 0) * delta_t
        dva_contrib = -(1 - RB) * prob_B_default * discount_factor(t_mid) * max(-ene, 0) * delta_t
        
        cva_contributions.append(cva_contrib)
        dva_contributions.append(dva_contrib)

    return cva_contributions, dva_contributions

# App layout
app.layout = html.Div([
    html.H1('Bilateral CVA Visualization - Replication Approach with Gaussian Copula', style={'text-align': 'center'}),
    
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
            
            html.Div([
                html.Label('Correlation (ρ, %):'),
                dcc.Input(id='rho-input', type='number', value=30.0, min=-100, max=100, step=1),
            ], style={'margin-bottom': '15px'}),
            
            html.Button('Calculate', id='calculate-button', n_clicks=0, 
                       style={'width': '100%', 'padding': '10px', 'margin-bottom': '20px'}),
            
            html.Hr(),
            html.H3('Notes'),
            html.P('This application implements bilateral CVA with a Gaussian Copula for dependence.'),
            html.P('CVA = -(1-RC) * P(C defaults, B survives) * DF(t) * EPE(t)'),
            html.P('DVA = -(1-RB) * P(B defaults, C survives) * DF(t) * ENE(t)'),
            html.P('Dependence modeled via Gaussian Copula: C(u,v) = Ψ(Ψ⁻¹(u), Ψ⁻¹(v); ρ)'),
            html.P('ρ represents correlation between default times (0% = independent).'),
            html.P('Warning: Gaussian Copula may underestimate tail risk.')
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
     State('rb-input', 'value'),
     State('rho-input', 'value')]
)
def update_graphs(n_clicks, r, lambda_c, lambda_b, rc, rb, rho):
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
        rho = rho / 100 if rho is not None else 0.3

        # Calculate CVA and DVA with copula
        cva_contribs, dva_contribs = calculate_cva_dva(r, lambda_c, lambda_b, rc, rb, rho)
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