import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import integrate
from scipy.stats import norm, multivariate_normal

# Sample data for 3 counterparties with trade-level breakdown
counterparties = {
    'Counterparty A': {
        'Tenor': np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        'Trade1_EPE': np.array([0, 200000, 250000, 260000, 250000, 230000, 210000, 170000, 130000, 80000, 25000]),
        'Trade1_ENE': np.array([0, -250000, -320000, -380000, -420000, -440000, -430000, -390000, -320000, -210000, -80000]),
        'Trade2_EPE': np.array([0, 300000, 370000, 390000, 380000, 350000, 320000, 260000, 200000, 120000, 40000]),
        'Trade2_ENE': np.array([0, -350000, -480000, -570000, -620000, -650000, -640000, -580000, -480000, -320000, -120000]),
        'Trade3_EPE': np.array([0, 201997, 257137, 264128, 250804, 236664, 209458, 173905, 143606, 90466, 28556]),
        'Trade3_ENE': np.array([0, -190189, -338387, -408724, -443909, -462157, -453924, -409604, -348750, -238355, -86472])
    },
    'Counterparty B': {
        'Tenor': np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        'Trade1_EPE': np.array([0, 150000, 180000, 200000, 190000, 180000, 160000, 140000, 120000, 100000, 60000]),
        'Trade1_ENE': np.array([0, -130000, -160000, -180000, -170000, -160000, -150000, -130000, -110000, -90000, -50000]),
        'Trade2_EPE': np.array([0, 200000, 240000, 260000, 250000, 230000, 210000, 180000, 150000, 120000, 80000]),
        'Trade2_ENE': np.array([0, -180000, -220000, -240000, -230000, -210000, -190000, -170000, -140000, -110000, -70000]),
        'Trade3_EPE': np.array([0, 150000, 180000, 190000, 180000, 170000, 150000, 130000, 110000, 80000, 60000]),
        'Trade3_ENE': np.array([0, -140000, -170000, -180000, -170000, -160000, -140000, -130000, -110000, -80000, -60000])
    },
    'Counterparty C': {
        'Tenor': np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        'Trade1_EPE': np.array([0, 100000, 120000, 130000, 120000, 110000, 100000, 90000, 70000, 50000, 30000]),
        'Trade1_ENE': np.array([0, -110000, -130000, -140000, -130000, -120000, -110000, -100000, -80000, -60000, -40000]),
        'Trade2_EPE': np.array([0, 120000, 140000, 150000, 140000, 130000, 120000, 100000, 80000, 60000, 40000]),
        'Trade2_ENE': np.array([0, -130000, -150000, -160000, -150000, -140000, -130000, -110000, -90000, -70000, -50000]),
        'Trade3_EPE': np.array([0, 80000, 90000, 120000, 110000, 100000, 80000, 60000, 50000, 40000, 30000]),
        'Trade3_ENE': np.array([0, -90000, -100000, -110000, -100000, -90000, -80000, -60000, -50000, -40000, -30000])
    }
}

# Convert to DataFrames and aggregate
dfs = {}
for name, data in counterparties.items():
    df = pd.DataFrame(data)
    df['EPE'] = df[['Trade1_EPE', 'Trade2_EPE', 'Trade3_EPE']].sum(axis=1)
    df['ENE'] = df[['Trade1_ENE', 'Trade2_ENE', 'Trade3_ENE']].sum(axis=1)
    dfs[name] = df

# Initialize Dash app
app = dash.Dash(__name__)

# Bootstrap hazard rates from CDS spreads
def bootstrap_hazard_rates(cds_spreads, tenors, r, recovery_rate):
    dt = np.diff(tenors)  # 10 intervals from 11 tenors
    if len(cds_spreads) != len(dt):
        raise ValueError(f"Expected {len(dt)} CDS spreads, got {len(cds_spreads)}")
    
    hazard_rates = []
    survival_probs = [1.0]  # P(τ > 0) = 1
    
    for i, spread in enumerate(cds_spreads):
        if i == 0:
            h = spread / ((1 - recovery_rate) * (1 + r * dt[i]))
        else:
            cumulative_default = sum(hazard_rates[j] * survival_probs[j] * dt[j] for j in range(i))
            h = (spread - cumulative_default) / ((1 - recovery_rate) * survival_probs[i] * dt[i])
        hazard_rates.append(h)
        survival_probs.append(survival_probs[-1] * np.exp(-h * dt[i]))
    
    return np.array(hazard_rates)

# Calculation functions
def calculate_cva_dva(epe, ene, r, cds_spreads_C, cds_spreads_B, RC, RB, rho):
    tenors = epe.index
    hazard_rates_C = bootstrap_hazard_rates(cds_spreads_C, tenors, r, RC)
    hazard_rates_B = bootstrap_hazard_rates(cds_spreads_B, tenors, r, RB)

    def discount_factor(t):
        idx = np.searchsorted(tenors, t, side='right') - 1
        if idx < 0:
            return 1.0
        return np.exp(-integrate.quad(lambda u: r + hazard_rates_C[idx] + hazard_rates_B[idx], 0, t)[0])

    def survival_prob(t, hazard_rates):
        idx = np.searchsorted(tenors, t, side='right') - 1
        if idx < 0:
            return 1.0
        return np.exp(-integrate.quad(lambda u: hazard_rates[min(idx, len(hazard_rates)-1)], 0, t)[0])

    def gaussian_copula_survival(t1, t2, hazard_rates_C, hazard_rates_B, rho):
        u = survival_prob(t1, hazard_rates_C)
        v = survival_prob(t2, hazard_rates_B)
        z1 = norm.ppf(1 - u)
        z2 = norm.ppf(1 - v)
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        return multivariate_normal.cdf([z1, z2], mean=mean, cov=cov)

    cva_contributions = [0]
    dva_contributions = [0]

    for i in range(1, len(tenors)):
        t_start = tenors[i-1]
        t_end = tenors[i]
        t_mid = (t_start + t_end) / 2

        survival_C = survival_prob(t_end, hazard_rates_C)
        survival_B = survival_prob(t_end, hazard_rates_B)
        joint_survival = gaussian_copula_survival(t_end, t_end, hazard_rates_C, hazard_rates_B, rho)
        
        delta_t = t_end - t_start
        prob_C_default = (survival_prob(t_start, hazard_rates_C) - survival_C) * (survival_B / joint_survival if joint_survival > 0 else 1)
        prob_B_default = (survival_prob(t_start, hazard_rates_B) - survival_B) * (survival_C / joint_survival if joint_survival > 0 else 1)

        cva_contrib = -(1 - RC) * prob_C_default * discount_factor(t_mid) * max(epe[i], 0) * delta_t
        dva_contrib = -(1 - RB) * prob_B_default * discount_factor(t_mid) * max(-ene[i], 0) * delta_t
        
        cva_contributions.append(cva_contrib)
        dva_contributions.append(dva_contrib)

    return cva_contributions, dva_contributions

def allocate_cva_dva(df_trades, epe, ene, cva_total, dva_total, r, cds_spreads_C, cds_spreads_B, RC, RB, rho, delta=0.01):
    allocated_cva = []
    allocated_dva = []
    for i in range(3):
        trade_epe = df_trades[f'Trade{i+1}_EPE']
        trade_ene = df_trades[f'Trade{i+1}_ENE']
        
        perturbed_epe = epe + delta * trade_epe
        perturbed_ene = ene + delta * trade_ene
        
        perturbed_cva, perturbed_dva = calculate_cva_dva(perturbed_epe, perturbed_ene, r, cds_spreads_C, cds_spreads_B, RC, RB, rho)
        cva_sensitivity = (sum(perturbed_cva) - cva_total) / delta
        dva_sensitivity = (sum(perturbed_dva) - dva_total) / delta
        
        allocated_cva.append(cva_sensitivity)
        allocated_dva.append(dva_sensitivity)
    
    return allocated_cva, allocated_dva

# App layout
app.layout = html.Div([
    html.H1('Bilateral CVA - CDS Implied Probabilities', style={'text-align': 'center'}),
    
    html.Div([
        # Left sidebar
        html.Div([
            html.H3('Input Parameters'),
            html.Div([html.Label('Risk-free Rate (%):'), dcc.Input(id='r-input', type='number', value=2.0, min=0, max=100, step=0.1)], style={'margin-bottom': '15px'}),
            html.Div([html.Label('Counterparty CDS Spreads (bps, 10 values):'), dcc.Input(id='cds-c-input', type='text', value='300, 300, 300, 300, 300, 300, 300, 300, 300, 300')], style={'margin-bottom': '15px'}),
            html.Div([html.Label('Bank CDS Spreads (bps, 10 values):'), dcc.Input(id='cds-b-input', type='text', value='100, 100, 100, 100, 100, 100, 100, 100, 100, 100')], style={'margin-bottom': '15px'}),
            html.Div([html.Label('Counterparty Recovery Rate (%):'), dcc.Input(id='rc-input', type='number', value=40.0, min=0, max=100, step=1)], style={'margin-bottom': '15px'}),
            html.Div([html.Label('Bank Recovery Rate (%):'), dcc.Input(id='rb-input', type='number', value=40.0, min=0, max=100, step=1)], style={'margin-bottom': '15px'}),
            html.Div([html.Label('Correlation (ρ, %):'), dcc.Input(id='rho-input', type='number', value=30.0, min=-100, max=100, step=1)], style={'margin-bottom': '15px'}),
            html.Div([html.Label('New Trade EPE Multiplier:'), dcc.Input(id='new-trade-epe', type='number', value=1.0, min=0, step=0.1)], style={'margin-bottom': '15px'}),
            html.Div([html.Label('New Trade ENE Multiplier:'), dcc.Input(id='new-trade-ene', type='number', value=1.0, min=0, step=0.1)], style={'margin-bottom': '15px'}),
            html.Button('Calculate', id='calculate-button', n_clicks=0, style={'width': '100%', 'padding': '10px', 'margin-bottom': '20px'}),
            
            html.Hr(),
            html.H3('Notes'),
            html.P('Uses CDS spreads (10 values for 11 tenors) to derive hazard rates.'),
            html.P('Survival Prob: P(τ>T) = exp(-∫λ(s)ds)'),
            html.P('Incremental CVA/DVA for new trade.'),
            html.P('Allocated CVA/DVA per trade.'),
            html.P('Gaussian Copula with ρ for dependence.')
        ], style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '20px'}),
        
        # Main content
        html.Div([
            html.Div(id='results-output'),
            dcc.Dropdown(
                id='counterparty-dropdown',
                options=[{'label': name, 'value': name} for name in counterparties.keys()],
                value='Counterparty A',
                style={'width': '50%', 'margin-bottom': '20px'}
            ),
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
    [Input('calculate-button', 'n_clicks'),
     Input('counterparty-dropdown', 'value')],
    [State('r-input', 'value'),
     State('cds-c-input', 'value'),
     State('cds-b-input', 'value'),
     State('rc-input', 'value'),
     State('rb-input', 'value'),
     State('rho-input', 'value'),
     State('new-trade-epe', 'value'),
     State('new-trade-ene', 'value')]
)
def update_graphs(n_clicks, selected_counterparty, r, cds_c, cds_b, rc, rb, rho, new_epe_mult, new_ene_mult):
    fig1 = go.Figure().update_layout(title=f'Exposure Profiles - {selected_counterparty}', xaxis_title='Tenor (years)', yaxis_title='Exposure', template='plotly_white')
    fig2 = go.Figure().update_layout(title=f'Counterparty-Level CVA/DVA - {selected_counterparty}', xaxis_title='Tenor (years)', yaxis_title='Contribution', template='plotly_white')
    fig3 = go.Figure().update_layout(title=f'Allocated CVA/DVA - {selected_counterparty}', xaxis_title='Trade', yaxis_title='Amount', template='plotly_white')
    results = html.Div("Press Calculate to see results", style={'text-align': 'center'})

    if n_clicks > 0:
        try:
            # Convert inputs
            r = r / 100 if r is not None else 0.02
            cds_spreads_C = np.array([float(x.strip()) / 10000 for x in cds_c.split(',')]) if cds_c else np.full(10, 0.03)
            cds_spreads_B = np.array([float(x.strip()) / 10000 for x in cds_b.split(',')]) if cds_b else np.full(10, 0.01)
            rc = rc / 100 if rc is not None else 0.4
            rb = rb / 100 if rb is not None else 0.4
            rho = rho / 100 if rho is not None else 0.3
            new_epe_mult = new_epe_mult if new_epe_mult is not None else 1.0
            new_ene_mult = new_ene_mult if new_ene_mult is not None else 1.0

            # Get data for selected counterparty
            df_selected = dfs[selected_counterparty]
            
            # Original portfolio
            epe_orig = df_selected['EPE']
            ene_orig = df_selected['ENE']
            cva_orig, dva_orig = calculate_cva_dva(epe_orig, ene_orig, r, cds_spreads_C, cds_spreads_B, rc, rb, rho)
            total_cva_orig = sum(cva_orig)
            total_dva_orig = sum(dva_orig)

            # New trade (scaled version of Trade1)
            new_trade_epe = df_selected['Trade1_EPE'] * new_epe_mult
            new_trade_ene = df_selected['Trade1_ENE'] * new_ene_mult
            epe_new = epe_orig + new_trade_epe
            ene_new = ene_orig + new_trade_ene
            cva_new, dva_new = calculate_cva_dva(epe_new, ene_new, r, cds_spreads_C, cds_spreads_B, rc, rb, rho)
            total_cva_new = sum(cva_new)
            total_dva_new = sum(dva_new)

            # Incremental CVA/DVA
            incremental_cva = total_cva_new - total_cva_orig
            incremental_dva = total_dva_new - total_dva_orig

            # Allocated CVA/DVA
            allocated_cva, allocated_dva = allocate_cva_dva(df_selected, epe_orig, ene_orig, total_cva_orig, total_dva_orig, r, cds_spreads_C, cds_spreads_B, rc, rb, rho)

            # Exposure graph
            fig1.add_trace(go.Scatter(x=df_selected['Tenor'], y=epe_orig, name='Original EPE', mode='lines+markers'))
            fig1.add_trace(go.Scatter(x=df_selected['Tenor'], y=ene_orig, name='Original ENE', mode='lines+markers'))
            fig1.add_trace(go.Scatter(x=df_selected['Tenor'], y=epe_new, name='New EPE', mode='lines+markers', line=dict(dash='dash')))
            fig1.add_trace(go.Scatter(x=df_selected['Tenor'], y=ene_new, name='New ENE', mode='lines+markers', line=dict(dash='dash')))

            # CVA/DVA graph
            fig2.add_trace(go.Bar(x=df_selected['Tenor'], y=cva_new, name='CVA Contribution'))
            fig2.add_trace(go.Bar(x=df_selected['Tenor'], y=dva_new, name='DVA Contribution'))
            fig2.update_layout(barmode='relative')

            # Allocated CVA/DVA graph
            trades = ['Trade 1', 'Trade 2', 'Trade 3']
            fig3.add_trace(go.Bar(x=trades, y=allocated_cva, name='Allocated CVA'))
            fig3.add_trace(go.Bar(x=trades, y=allocated_dva, name='Allocated DVA'))
            fig3.update_layout(barmode='group')

            # Results
            results = html.Div([
                html.H3(f'Original BCVA ({selected_counterparty}): {total_cva_orig + total_dva_orig:,.2f}'),
                html.H3(f'New BCVA ({selected_counterparty}): {total_cva_new + total_dva_new:,.2f}'),
                html.H3(f'Incremental CVA: {incremental_cva:,.2f}'),
                html.H3(f'Incremental DVA: {incremental_dva:,.2f}'),
                html.H3(f'Allocated CVA Sum: {sum(allocated_cva):,.2f}'),
                html.H3(f'Allocated DVA Sum: {sum(allocated_dva):,.2f}'),
            ], style={'text-align': 'center'})

        except ValueError as e:
            results = html.Div(f"Error: {str(e)}", style={'text-align': 'center', 'color': 'red'})
        except Exception as e:
            results = html.Div(f"Unexpected error: {str(e)}", style={'text-align': 'center', 'color': 'red'})

    return fig1, fig2, fig3, results

# Run the app
if __name__ == '__main__':
    app.run(debug=True)