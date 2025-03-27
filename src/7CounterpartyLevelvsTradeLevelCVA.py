import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import integrate
from scipy.stats import norm, multivariate_normal

# Initial netting set data (3 trades)
trades_data = {
    'Tenor': np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
    'Trade1_EPE': np.array([0, 200000, 250000, 260000, 250000, 230000, 210000, 170000, 130000, 80000, 25000]),
    'Trade1_ENE': np.array([0, -250000, -320000, -380000, -420000, -440000, -430000, -390000, -320000, -210000, -80000]),
    'Trade2_EPE': np.array([0, 300000, 370000, 390000, 380000, 350000, 320000, 260000, 200000, 120000, 40000]),
    'Trade2_ENE': np.array([0, -350000, -480000, -570000, -620000, -650000, -640000, -580000, -480000, -320000, -120000]),
    'Trade3_EPE': np.array([0, 201997, 257137, 264128, 250804, 236664, 209458, 173905, 143606, 90466, 28556]),
    'Trade3_ENE': np.array([0, -190189, -338387, -408724, -443909, -462157, -453924, -409604, -348750, -238355, -86472])
}
df_trades = pd.DataFrame(trades_data)

# Aggregate to netting set level
df = pd.DataFrame({
    'Tenor': df_trades['Tenor'],
    'EPE': df_trades[['Trade1_EPE', 'Trade2_EPE', 'Trade3_EPE']].sum(axis=1),
    'ENE': df_trades[['Trade1_ENE', 'Trade2_ENE', 'Trade3_ENE']].sum(axis=1)
})

# Initialize Dash app
app = dash.Dash(__name__)

# Calculation functions
def calculate_cva_dva(epe, ene, r, lambda_C, lambda_B, RC, RB, rho):
    def discount_factor(t):
        return np.exp(-integrate.quad(lambda u: r + lambda_C + lambda_B, 0, t)[0])

    def survival_prob(t, lambda_):
        return np.exp(-lambda_ * t)

    def gaussian_copula_survival(t1, t2, lambda_C, lambda_B, rho):
        u = survival_prob(t1, lambda_C)
        v = survival_prob(t2, lambda_B)
        z1 = norm.ppf(1 - u)
        z2 = norm.ppf(1 - v)
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        return multivariate_normal.cdf([z1, z2], mean=mean, cov=cov)

    tenors = df['Tenor']
    cva_contributions = [0]
    dva_contributions = [0]

    for i in range(1, len(tenors)):
        t_start = tenors[i-1]
        t_end = tenors[i]
        t_mid = (t_start + t_end) / 2

        survival_C = survival_prob(t_end, lambda_C)
        survival_B = survival_prob(t_end, lambda_B)
        joint_survival = gaussian_copula_survival(t_end, t_end, lambda_C, lambda_B, rho)
        
        delta_t = t_end - t_start
        prob_C_default = (survival_prob(t_start, lambda_C) - survival_C) * (survival_B / joint_survival if joint_survival > 0 else 1)
        prob_B_default = (survival_prob(t_start, lambda_B) - survival_B) * (survival_C / joint_survival if joint_survival > 0 else 1)

        cva_contrib = -(1 - RC) * prob_C_default * discount_factor(t_mid) * max(epe[i], 0) * delta_t
        dva_contrib = -(1 - RB) * prob_B_default * discount_factor(t_mid) * max(-ene[i], 0) * delta_t
        
        cva_contributions.append(cva_contrib)
        dva_contributions.append(dva_contrib)

    return cva_contributions, dva_contributions

def allocate_cva_dva(epe, ene, cva_total, dva_total, r, lambda_C, lambda_B, RC, RB, rho, delta=0.01):
    allocated_cva = []
    allocated_dva = []
    for i in range(3):  # 3 trades
        trade_epe = df_trades[f'Trade{i+1}_EPE']
        trade_ene = df_trades[f'Trade{i+1}_ENE']
        
        # Perturbed EPE/ENE
        perturbed_epe = epe + delta * trade_epe
        perturbed_ene = ene + delta * trade_ene
        
        perturbed_cva, perturbed_dva = calculate_cva_dva(perturbed_epe, perturbed_ene, r, lambda_C, lambda_B, RC, RB, rho)
        cva_sensitivity = (sum(perturbed_cva) - cva_total) / delta
        dva_sensitivity = (sum(perturbed_dva) - dva_total) / delta
        
        allocated_cva.append(cva_sensitivity)
        allocated_dva.append(dva_sensitivity)
    
    return allocated_cva, allocated_dva

# App layout
app.layout = html.Div([
    html.H1('Bilateral CVA - Counterparty vs Trade Level', style={'text-align': 'center'}),
    
    html.Div([
        # Left sidebar
        html.Div([
            html.H3('Input Parameters'),
            
            html.Div([html.Label('Risk-free Rate (%):'), dcc.Input(id='r-input', type='number', value=2.0, min=0, max=100, step=0.1)], style={'margin-bottom': '15px'}),
            html.Div([html.Label('Counterparty Hazard Rate (%):'), dcc.Input(id='lambda-c-input', type='number', value=3.0, min=0, max=100, step=0.1)], style={'margin-bottom': '15px'}),
            html.Div([html.Label('Bank Hazard Rate (%):'), dcc.Input(id='lambda-b-input', type='number', value=1.0, min=0, max=100, step=0.1)], style={'margin-bottom': '15px'}),
            html.Div([html.Label('Counterparty Recovery Rate (%):'), dcc.Input(id='rc-input', type='number', value=40.0, min=0, max=100, step=1)], style={'margin-bottom': '15px'}),
            html.Div([html.Label('Bank Recovery Rate (%):'), dcc.Input(id='rb-input', type='number', value=40.0, min=0, max=100, step=1)], style={'margin-bottom': '15px'}),
            html.Div([html.Label('Correlation (ρ, %):'), dcc.Input(id='rho-input', type='number', value=30.0, min=-100, max=100, step=1)], style={'margin-bottom': '15px'}),
            
            html.Div([
                html.Label('New Trade EPE Multiplier:'),
                dcc.Input(id='new-trade-epe', type='number', value=1.0, min=0, step=0.1),
            ], style={'margin-bottom': '15px'}),
            
            html.Div([
                html.Label('New Trade ENE Multiplier:'),
                dcc.Input(id='new-trade-ene', type='number', value=1.0, min=0, step=0.1),
            ], style={'margin-bottom': '15px'}),
            
            html.Button('Calculate', id='calculate-button', n_clicks=0, style={'width': '100%', 'padding': '10px', 'margin-bottom': '20px'}),
            
            html.Hr(),
            html.H3('Notes'),
            html.P('Counterparty-level CVA uses netted exposures.'),
            html.P('Incremental CVA shows impact of a new trade.'),
            html.P('Allocated CVA breaks total to trade level using Euler method.'),
            html.P('Gaussian Copula models dependence with ρ.'),
            html.P('Warning: Gaussian Copula may underestimate tail risk.')
        ], style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '20px'}),
        
        # Main content
        html.Div([
            html.Div(id='results-output'),
            dcc.Graph(id='exposure-graph'),
            dcc.Graph(id='cva-dva-graph'),
            dcc.Graph(id='allocated-graph'),
        ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'})
    ])
])

# Callback to update graphs and results
@app.callback(
    [Output('exposure-graph', 'figure'),
     Output('cva-dva-graph', 'figure'),
     Output('allocated-graph', 'figure'),
     Output('results-output', 'children')],
    [Input('calculate-button', 'n_clicks')],
    [State('r-input', 'value'),
     State('lambda-c-input', 'value'),
     State('lambda-b-input', 'value'),
     State('rc-input', 'value'),
     State('rb-input', 'value'),
     State('rho-input', 'value'),
     State('new-trade-epe', 'value'),
     State('new-trade-ene', 'value')]
)
def update_graphs(n_clicks, r, lambda_c, lambda_b, rc, rb, rho, new_epe_mult, new_ene_mult):
    fig1 = go.Figure().update_layout(title='Expected Exposure Profiles', xaxis_title='Tenor (years)', yaxis_title='Exposure', template='plotly_white')
    fig2 = go.Figure().update_layout(title='Counterparty-Level CVA/DVA', xaxis_title='Tenor (years)', yaxis_title='Contribution', template='plotly_white')
    fig3 = go.Figure().update_layout(title='Allocated CVA/DVA by Trade', xaxis_title='Trade', yaxis_title='Amount', template='plotly_white')
    results = html.Div("Press Calculate to see results", style={'text-align': 'center'})

    if n_clicks > 0:
        # Convert inputs
        r = r / 100 if r is not None else 0.02
        lambda_c = lambda_c / 100 if lambda_c is not None else 0.03
        lambda_b = lambda_b / 100 if lambda_b is not None else 0.01
        rc = rc / 100 if rc is not None else 0.4
        rb = rb / 100 if rb is not None else 0.4
        rho = rho / 100 if rho is not None else 0.3
        new_epe_mult = new_epe_mult if new_epe_mult is not None else 1.0
        new_ene_mult = new_ene_mult if new_ene_mult is not None else 1.0

        # Original portfolio
        epe_orig = df['EPE']
        ene_orig = df['ENE']
        cva_orig, dva_orig = calculate_cva_dva(epe_orig, ene_orig, r, lambda_c, lambda_b, rc, rb, rho)
        total_cva_orig = sum(cva_orig)
        total_dva_orig = sum(dva_orig)

        # New trade (scaled version of Trade1 as example)
        new_trade_epe = df_trades['Trade1_EPE'] * new_epe_mult
        new_trade_ene = df_trades['Trade1_ENE'] * new_ene_mult
        epe_new = epe_orig + new_trade_epe
        ene_new = ene_orig + new_trade_ene
        cva_new, dva_new = calculate_cva_dva(epe_new, ene_new, r, lambda_c, lambda_b, rc, rb, rho)
        total_cva_new = sum(cva_new)
        total_dva_new = sum(dva_new)

        # Incremental CVA/DVA
        incremental_cva = total_cva_new - total_cva_orig
        incremental_dva = total_dva_new - total_dva_orig

        # Allocated CVA/DVA
        allocated_cva, allocated_dva = allocate_cva_dva(epe_orig, ene_orig, total_cva_orig, total_dva_orig, r, lambda_c, lambda_b, rc, rb, rho)

        # Exposure graph
        fig1.add_trace(go.Scatter(x=df['Tenor'], y=epe_orig, name='Original EPE', mode='lines+markers'))
        fig1.add_trace(go.Scatter(x=df['Tenor'], y=ene_orig, name='Original ENE', mode='lines+markers'))
        fig1.add_trace(go.Scatter(x=df['Tenor'], y=epe_new, name='New EPE', mode='lines+markers', line=dict(dash='dash')))
        fig1.add_trace(go.Scatter(x=df['Tenor'], y=ene_new, name='New ENE', mode='lines+markers', line=dict(dash='dash')))

        # CVA/DVA graph (counterparty level)
        fig2.add_trace(go.Bar(x=df['Tenor'], y=cva_new, name='CVA Contribution'))
        fig2.add_trace(go.Bar(x=df['Tenor'], y=dva_new, name='DVA Contribution'))
        fig2.update_layout(barmode='relative')

        # Allocated CVA/DVA graph
        trades = ['Trade 1', 'Trade 2', 'Trade 3']
        fig3.add_trace(go.Bar(x=trades, y=allocated_cva, name='Allocated CVA'))
        fig3.add_trace(go.Bar(x=trades, y=allocated_dva, name='Allocated DVA'))
        fig3.update_layout(barmode='group')

        # Results
        results = html.Div([
            html.H3(f'Original BCVA: {total_cva_orig + total_dva_orig:,.2f}'),
            html.H3(f'New BCVA: {total_cva_new + total_dva_new:,.2f}'),
            html.H3(f'Incremental CVA: {incremental_cva:,.2f}'),
            html.H3(f'Incremental DVA: {incremental_dva:,.2f}'),
            html.H3(f'Allocated CVA Sum: {sum(allocated_cva):,.2f}'),
            html.H3(f'Allocated DVA Sum: {sum(allocated_dva):,.2f}'),
        ], style={'text-align': 'center'})

    return fig1, fig2, fig3, results

# Run the app
if __name__ == '__main__':
    app.run(debug=True)