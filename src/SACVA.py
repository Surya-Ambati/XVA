import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.interpolate
from datetime import datetime, timedelta
import os

# Constants
NO_OF_EXPOSURE_DATES = 90
HURDLE_RATE = 0.14625
MASTER_FOLDER = os.getcwd() + "\\"
TENORS = ['6M', '1Y', '3Y', '5Y', '10Y']
BUCKET_PERIOD = np.array([0.5, 1, 3, 5, 7, 10])  # Years
CROSS_BUCKET_CORR_MAT = np.array([
    [0.0, 10.0, 20.0, 25.0, 20.0, 15.0, 10, 0.0, 45.0],
    [10.0, 0.0, 5.0, 15.0, 20.0, 5.0, 20, 0.0, 45.0],
    [20.0, 5.0, 0.0, 20.0, 25.0, 5.0, 5, 0.0, 45.0],
    [25.0, 15.0, 20.0, 0.0, 25.0, 5.0, 15, 0.0, 45.0],
    [20.0, 20.0, 25.0, 25.0, 0.0, 5.0, 20, 0.0, 45.0],
    [15.0, 5.0, 5.0, 5.0, 5.0, 0.0, 5, 0.0, 45.0],
    [10, 20, 5, 15, 20, 5, 0, 0.0, 45.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0],
    [45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 0.0, 0.0]
]) / 100
MULT = 1

def generate_dummy_data():
    today = datetime.today()
    exposure_dates = [today + timedelta(days=i) for i in range(NO_OF_EXPOSURE_DATES)]
    exposure_dates_serial = [(d - pd.Timestamp('1899-12-30')).days for d in exposure_dates]
    
    grid_ids = [1001, 1002, 1003, 1004, 1005]
    np.random.seed(42)
    epe_data = {grid_id: np.random.lognormal(mean=10, sigma=1, size=NO_OF_EXPOSURE_DATES) for grid_id in grid_ids}
    epe_df = pd.DataFrame(epe_data, index=exposure_dates_serial)
    
    grid_to_curve = {1001: ['CurveA'], 1002: ['CurveB'], 1003: ['CurveC'], 1004: ['CurveD'], 1005: ['CurveE']}
    curve_to_rr = {'CurveA': 0.4, 'CurveB': 0.4, 'CurveC': 0.4, 'CurveD': 0.4, 'CurveE': 0.4}
    
    tenor_points = [0, 0.5, 1, 3, 5, 10]
    hazard_curves = {
        'CurveA': np.array([tenor_points, [0.01]*6]).T,
        'CurveB': np.array([tenor_points, [0.02]*6]).T,
        'CurveC': np.array([tenor_points, [0.015]*6]).T,
        'CurveD': np.array([tenor_points, [0.025]*6]).T,
        'CurveE': np.array([tenor_points, [0.03]*6]).T,
    }
    
    ir_curve = np.array([[0, 0.01], [1, 0.01], [2, 0.01], [5, 0.01], [10, 0.01], [20, 0.01]])
    ir_interpolator = scipy.interpolate.interp1d(ir_curve[:, 0], ir_curve[:, 1], fill_value="extrapolate")
    
    sa_cva_report = pd.DataFrame({
        'Bucket': [b for b in range(1, 6) for _ in TENORS],
        'Tenor': TENORS * 5,
        'CptyGrid': [grid_id for grid_id in grid_ids for _ in TENORS],
        'CptyParentGrid': [grid_id for grid_id in grid_ids for _ in TENORS],
        'CreditQuality': ['CQS1'] * (5 * len(TENORS)),
        'RW': [0.007, 0.008, 0.01, 0.0135, 0.0225] * 5
    })
    
    sp = np.linspace(1, 0.75, NO_OF_EXPOSURE_DATES)  # Dummy survival probability
    
    return {
        'exposure_dates': exposure_dates,
        'exposure_dates_serial': exposure_dates_serial,
        'epe_df': epe_df,
        'grid_to_curve': grid_to_curve,
        'curve_to_rr': curve_to_rr,
        'hazard_curves': hazard_curves,
        'ir_interpolator': ir_interpolator,
        'sa_cva_report': sa_cva_report,
        'sp': sp
    }

data = generate_dummy_data()

def compute_survival_probabilities(hazard_curve, times):
    hazard_interp = scipy.interpolate.interp1d(hazard_curve[:, 0], hazard_curve[:, 1], kind='previous', fill_value="extrapolate")
    cum_hazard = np.zeros_like(times)
    for i in range(1, len(times)):
        t_prev, t_curr = times[i-1], times[i]
        hazard_rate = hazard_interp((t_prev + t_curr) / 2)
        cum_hazard[i] = cum_hazard[i-1] + hazard_rate * (t_curr - t_prev)
    return np.exp(-cum_hazard)

def compute_cva_from_time(epe, hazard_curve, R, times, start_idx):
    survival_probs = compute_survival_probabilities(hazard_curve, times[start_idx:])
    pd_diff = survival_probs[:-1] - survival_probs[1:]
    epe_slice = epe[start_idx+1:]
    return (1 - R) * np.sum(epe_slice * pd_diff)

def compute_bucketed_sensitivities(epe, hazard_curve, R, times, exp_i, bump_size=0.0001):
    base_cva = compute_cva_from_time(epe, hazard_curve, R, times, exp_i)
    n_buckets = len(hazard_curve) - 1
    sensitivities = []
    for j in range(n_buckets):
        bumped_curve = hazard_curve.copy()
        bumped_curve[j, 1] += bump_size
        bumped_cva = compute_cva_from_time(epe, bumped_curve, R, times, exp_i)
        sensi = (bumped_cva - base_cva) / bump_size
        sensitivities.append(sensi)
    return np.array(sensitivities)

def compute_jacobian(hazard_curve, R, standard_tenors, bump_size=0.0001):
    n = len(standard_tenors)
    J = np.zeros((n, n))
    base_spreads = [compute_spread(hazard_curve, R, T) for T in standard_tenors]
    for j in range(n):
        bumped_curve = hazard_curve.copy()
        bumped_curve[j, 1] += bump_size
        bumped_spreads = [compute_spread(bumped_curve, R, T) for T in standard_tenors]
        J[:, j] = (np.array(bumped_spreads) - np.array(base_spreads)) / bump_size
    return J

def compute_spread(hazard_curve, R, T):
    times = hazard_curve[:, 0]
    hazards = hazard_curve[:, 1]
    idx = np.searchsorted(times, T, side='right')
    if idx == 0:
        return 0
    integral_prot = 0
    integral_prem = 0
    for k in range(idx):
        t_start = times[k-1] if k > 0 else 0
        t_end = min(times[k], T) if k < idx-1 else T
        lambda_k = hazards[k]
        S_start = np.exp(-sum(hazards[:k] * np.diff(times[:k+1])) if k > 0 else 0)
        delta_t = t_end - t_start
        integral_prot += lambda_k * S_start * (1 - np.exp(-lambda_k * delta_t)) / lambda_k
        integral_prem += S_start * (1 - np.exp(-lambda_k * delta_t)) / lambda_k
    prot_leg = (1 - R) * integral_prot
    prem_leg = integral_prem
    return prot_leg / prem_leg if prem_leg != 0 else 0

def compute_sensitivities(data):
    times = np.array(data['exposure_dates_serial']) / 365.25
    sensi_data = {}
    for grid_id in data['epe_df'].columns:
        curve_name = data['grid_to_curve'][grid_id][0]
        hazard_curve = data['hazard_curves'][curve_name]
        R = data['curve_to_rr'][curve_name]
        epe = data['epe_df'][grid_id].values
        sensi_matrix = np.zeros((len(TENORS), NO_OF_EXPOSURE_DATES))
        J = compute_jacobian(hazard_curve, R, [0.5, 1, 3, 5, 10])
        for exp_i in range(NO_OF_EXPOSURE_DATES):
            bucket_sensi = compute_bucketed_sensitivities(epe, hazard_curve, R, times, exp_i)
            tenor_sensi = np.linalg.solve(J, bucket_sensi[:5])  # Assuming 5 buckets align with tenors
            sensi_matrix[:, exp_i] = tenor_sensi
        sensi_data[grid_id] = pd.DataFrame(sensi_matrix, index=TENORS, columns=data['exposure_dates_serial'])
    return sensi_data

def build_corr_matrix(df_bucket):
    n = len(df_bucket)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if df_bucket.iloc[i]['grid'] == df_bucket.iloc[j]['grid'] and df_bucket.iloc[i]['tenor'] == df_bucket.iloc[j]['tenor']:
                corr = 1.0
            elif df_bucket.iloc[i]['grid'] == df_bucket.iloc[j]['grid']:
                corr = 0.9
            else:
                corr = 0.4
            corr_matrix[i, j] = corr_matrix[j, i] = corr
    return corr_matrix

def compute_bucket_capital(df_bucket):
    if df_bucket.empty:
        return {'capital_bucket_unhedged': 0, 'sum_ws_k_unhedged': 0, 'capital_bucket_hedged': 0, 'sum_ws_k_hedged': 0}
    WS = df_bucket['cva_s x RW'].values
    corr_matrix = build_corr_matrix(df_bucket)
    s_unhedged = WS @ corr_matrix @ WS
    K_b_unhedged = np.sqrt(max(s_unhedged, 0))
    sum_ws_k_unhedged = WS.sum()
    return {
        'capital_bucket_unhedged': K_b_unhedged,
        'sum_ws_k_unhedged': sum_ws_k_unhedged,
        'capital_bucket_hedged': K_b_unhedged,  # Assuming no hedging
        'sum_ws_k_hedged': sum_ws_k_unhedged
    }

def compute_capital_profile(sensi_data, sa_cva_report, dates):
    capital_profile = []
    for date in dates:
        sensi_df = pd.DataFrame({
            'grid': [grid_id for grid_id in sensi_data.keys() for _ in TENORS],
            'tenor': TENORS * len(sensi_data),
            'cva_s': [sensi_data[grid_id].loc[tenor, date] for grid_id in sensi_data.keys() for tenor in TENORS],
            'cva_h': 0.0
        })
        merged_df = pd.merge(sensi_df, sa_cva_report.rename(columns={'CptyGrid': 'grid', 'Tenor': 'tenor'}), on=['grid', 'tenor'])
        merged_df['cva_net'] = merged_df['cva_s'] + merged_df['cva_h']
        merged_df['cva_s x RW'] = merged_df['cva_s'] * merged_df['RW']
        merged_df['cva_net x RW'] = merged_df['cva_net'] * merged_df['RW']
        
        out_df = pd.DataFrame(columns=['bucket', 'capital_bucket_unhedged', 'sum_ws_k_unhedged', 'capital_bucket_hedged', 'sum_ws_k_hedged'])
        for bucket in range(1, 6):
            df_bucket = merged_df[merged_df['Bucket'] == bucket]
            dfff = compute_bucket_capital(df_bucket)
            out_df.loc[bucket-1] = [bucket, dfff['capital_bucket_unhedged'], dfff['sum_ws_k_unhedged'], dfff['capital_bucket_hedged'], dfff['sum_ws_k_hedged']]
        
        final_df = out_df.copy()
        final_df['s_b_unhedged'] = [max(-k, min(k, ws)) for k, ws in zip(final_df['capital_bucket_unhedged'], final_df['sum_ws_k_unhedged'])]
        K_b_unhedged2 = final_df['capital_bucket_unhedged']**2
        s_b_unhedged = final_df['s_b_unhedged']
        cap_unhedged = MULT * np.sqrt(sum(K_b_unhedged2) + s_b_unhedged @ CROSS_BUCKET_CORR_MAT[:5, :5] @ s_b_unhedged)
        capital_profile.append(cap_unhedged)
    return pd.DataFrame({'t': dates, 'cap': capital_profile})

def compute_cost(capital_profile):
    times = np.array([0] + [(d - data['exposure_dates_serial'][0]) / 365.25 for d in data['exposure_dates_serial']])
    capital = np.array([capital_profile['cap'].iloc[0]] + list(capital_profile['cap']))
    cost = 0
    for j in range(len(times)-1):
        cost += capital[j] * HURDLE_RATE * (times[j+1] - times[j])
    return cost


import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.interpolate
from datetime import datetime, timedelta

# Constants
NO_OF_EXPOSURE_DATES = 90
HURDLE_RATE = 0.14625
TENORS = ['6M', '1Y', '3Y', '5Y', '10Y']
BUCKET_PERIOD = np.array([0.5, 1, 3, 5, 7, 10])  # Years
CROSS_BUCKET_CORR_MAT = np.array([
    [0.0, 10.0, 20.0, 25.0, 20.0, 15.0, 10, 0.0, 45.0],
    [10.0, 0.0, 5.0, 15.0, 20.0, 5.0, 20, 0.0, 45.0],
    [20.0, 5.0, 0.0, 20.0, 25.0, 5.0, 5, 0.0, 45.0],
    [25.0, 15.0, 20.0, 0.0, 25.0, 5.0, 15, 0.0, 45.0],
    [20.0, 20.0, 25.0, 25.0, 0.0, 5.0, 20, 0.0, 45.0],
    [15.0, 5.0, 5.0, 5.0, 5.0, 0.0, 5, 0.0, 45.0],
    [10, 20, 5, 15, 20, 5, 0, 0.0, 45.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0],
    [45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 0.0, 0.0]
]) / 100
MULT = 1

# Generate Dummy Data
def generate_dummy_data():
    today = datetime.today()
    exposure_dates = [today + timedelta(days=i) for i in range(NO_OF_EXPOSURE_DATES)]
    exposure_dates_serial = [(d - pd.Timestamp('1899-12-30')).days for d in exposure_dates]
    
    grid_ids = [1001, 1002, 1003, 1004, 1005]
    np.random.seed(42)
    epe_data = {grid_id: np.random.lognormal(mean=10, sigma=1, size=NO_OF_EXPOSURE_DATES) for grid_id in grid_ids}
    epe_df = pd.DataFrame(epe_data, index=exposure_dates_serial)
    
    grid_to_curve = {1001: ['CurveA'], 1002: ['CurveB'], 1003: ['CurveC'], 1004: ['CurveD'], 1005: ['CurveE']}
    curve_to_rr = {'CurveA': 0.4, 'CurveB': 0.4, 'CurveC': 0.4, 'CurveD': 0.4, 'CurveE': 0.4}
    
    tenor_points = [0, 0.5, 1, 3, 5, 10]
    hazard_curves = {
        'CurveA': np.array([tenor_points, [0.01]*6]).T,
        'CurveB': np.array([tenor_points, [0.02]*6]).T,
        'CurveC': np.array([tenor_points, [0.015]*6]).T,
        'CurveD': np.array([tenor_points, [0.025]*6]).T,
        'CurveE': np.array([tenor_points, [0.03]*6]).T,
    }
    
    ir_curve = np.array([[0, 0.01], [1, 0.01], [2, 0.01], [5, 0.01], [10, 0.01], [20, 0.01]])
    ir_interpolator = scipy.interpolate.interp1d(ir_curve[:, 0], ir_curve[:, 1], fill_value="extrapolate")
    
    sa_cva_report = pd.DataFrame({
        'Bucket': [b for b in range(1, 6) for _ in TENORS],
        'Tenor': TENORS * 5,
        'CptyGrid': [grid_id for grid_id in grid_ids for _ in TENORS],
        'CptyParentGrid': [grid_id for grid_id in grid_ids for _ in TENORS],
        'CreditQuality': ['CQS1'] * (5 * len(TENORS)),
        'RW': [0.007, 0.008, 0.01, 0.0135, 0.0225] * 5
    })
    
    sp = np.linspace(1, 0.75, NO_OF_EXPOSURE_DATES)
    
    return {
        'exposure_dates': exposure_dates,
        'exposure_dates_serial': exposure_dates_serial,
        'epe_df': epe_df,
        'grid_to_curve': grid_to_curve,
        'curve_to_rr': curve_to_rr,
        'hazard_curves': hazard_curves,
        'ir_interpolator': ir_interpolator,
        'sa_cva_report': sa_cva_report,
        'sp': sp
    }

data = generate_dummy_data()

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("FRTB SA CVA Capital Cost Calculator", style={'text-align': 'center'}),
    
    # Store for results
    dcc.Store(id='result-store', data=[]),
    
    # Tabs
    dcc.Tabs([
        # Calculator Tab
        dcc.Tab(label='Calculator', children=[
            html.Div([
                # Instructions Section
                html.H3("Instructions", style={'margin-top': '20px'}),
                html.P("1. EOD: Computes capital profile for the entire portfolio. No input needed."),
                html.P("2. Standalone Counterparty: Enter a Grid ID (e.g., 1001) to compute for a specific counterparty."),
                html.P("3. Incremental Counterparty: Enter a Grid ID (e.g., 1002) to compute incremental impact."),
                html.P("4. Incremental Trade: Enter Grid ID and Run ID (e.g., 1001,123) to compute trade impact."),
                html.P("Valid Grid IDs: 1001, 1002, 1003, 1004, 1005"),
                html.P("Interpretation: The graph shows capital requirements over time. X-axis is dates, Y-axis is capital in arbitrary units. Results are saved in the 'Results' tab."),
            ], style={'padding': '10px', 'background-color': '#f0f0f0'}),
            
            # Input and Button Section
            html.Div([
                html.Button("EOD - EndOfDay Computation", id="btn-eod", n_clicks=0, style={'width': '100%', 'margin-bottom': '10px'}),
                
                html.Label("Standalone Counterparty (Grid ID):", style={'margin-top': '20px'}),
                dcc.Input(id="standalone-input", type="text", placeholder="e.g., 1001", style={'width': '100%'}),
                html.Button("Additional Capital Profile for Specific Cpty", id="btn-standalone", n_clicks=0, style={'width': '100%', 'margin-top': '5px'}),
                
                html.Label("Incremental Counterparty (Grid ID):", style={'margin-top': '20px'}),
                dcc.Input(id="incremental-cpty-input", type="text", placeholder="e.g., 1002", style={'width': '100%'}),
                html.Button("Capital Impact of adding a new Cpty", id="btn-incremental-cpty", n_clicks=0, style={'width': '100%', 'margin-top': '5px'}),
                
                html.Label("Incremental Trade (Grid ID, Run ID):", style={'margin-top': '20px'}),
                dcc.Input(id="incremental-trade-input", type="text", placeholder="e.g., 1001,123", style={'width': '100%'}),
                html.Button("Capital Impact of adding new trade for a Cpty", id="btn-incremental-trade", n_clicks=0, style={'width': '100%', 'margin-top': '5px'}),
                
                html.Div(id="status", children="Status: Ready", style={'margin-top': '20px'}),
            ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '20px'}),
            
            # Graph Section
            html.Div([
                dcc.Graph(id="capital-profile"),
            ], style={'width': '70%', 'display': 'inline-block', 'padding': '20px'}),
        ]),
        
        # Dummy Data Tab
        dcc.Tab(label='Dummy Data', children=[
            html.H3("Dummy Data Used for Calculations", style={'margin-top': '20px'}),
            html.P("This tab shows all the dummy data used in the calculations. Below are the datasets and their roles:"),
            html.Ul([
                html.Li("Exposure Profile (EPE): Contains the Expected Positive Exposure for each counterparty (Grid ID) over time. Used to compute CVA sensitivities."),
                html.Li("SA CVA Report: Static data defining buckets, tenors, and risk weights for each counterparty. Used to compute capital profiles."),
                html.Li("Grid to Curve Mapping: Maps each Grid ID to a hazard curve for CVA calculations."),
                html.Li("Curve to Recovery Rate: Defines the recovery rate for each curve, used in CVA calculations."),
                html.Li("Hazard Curves: Define the hazard rates over time for each curve, used to compute survival probabilities and CVA."),
                html.Li("IR Curve: Interest rate curve for discounting, used in CVA and cost calculations."),
                html.Li("SP: Survival probabilities, used in cost calculations.")
            ]),
            
            html.H4("Exposure Profile (EPE)", style={'margin-top': '20px'}),
            dash_table.DataTable(
                id='epe-table',
                columns=[{"name": "Date", "id": "Date"}] + [{"name": str(grid_id), "id": str(grid_id)} for grid_id in data['epe_df'].columns],
                data=[
                    {"Date": data['exposure_dates'][i].strftime('%Y-%m-%d'), **{str(grid_id): data['epe_df'][grid_id].iloc[i] for grid_id in data['epe_df'].columns}}
                    for i in range(NO_OF_EXPOSURE_DATES)
                ],
                style_table={'overflowX': 'auto'},
                page_size=10,
            ),
            
            html.H4("SA CVA Report", style={'margin-top': '20px'}),
            html.P("Buckets: Represent risk categories (1-5) based on counterparty type (e.g., financial, corporate). Tenors: Time horizons for sensitivities. RW: Risk weights used to scale sensitivities for capital calculation."),
            dash_table.DataTable(
                id='sa-cva-table',
                columns=[{"name": col, "id": col} for col in data['sa_cva_report'].columns],
                data=data['sa_cva_report'].to_dict('records'),
                style_table={'overflowX': 'auto'},
                page_size=10,
            ),
            
            html.H4("Grid to Curve Mapping", style={'margin-top': '20px'}),
            dash_table.DataTable(
                id='grid-to-curve-table',
                columns=[{"name": "Grid ID", "id": "Grid ID"}, {"name": "Curve", "id": "Curve"}],
                data=[{"Grid ID": k, "Curve": v[0]} for k, v in data['grid_to_curve'].items()],
                style_table={'overflowX': 'auto'},
                page_size=10,
            ),
            
            html.H4("Curve to Recovery Rate", style={'margin-top': '20px'}),
            dash_table.DataTable(
                id='curve-to-rr-table',
                columns=[{"name": "Curve", "id": "Curve"}, {"name": "Recovery Rate", "id": "Recovery Rate"}],
                data=[{"Curve": k, "Recovery Rate": v} for k, v in data['curve_to_rr'].items()],
                style_table={'overflowX': 'auto'},
                page_size=10,
            ),
            
            html.H4("Hazard Curves", style={'margin-top': '20px'}),
            html.P("Each curve has hazard rates over tenors (0, 0.5, 1, 3, 5, 10 years)."),
            dash_table.DataTable(
                id='hazard-curves-table',
                columns=[{"name": "Curve", "id": "Curve"}] + [{"name": f"Tenor {t}Y", "id": f"Tenor {t}Y"} for t in [0, 0.5, 1, 3, 5, 10]],
                data=[
                    {"Curve": curve, **{f"Tenor {t}Y": data['hazard_curves'][curve][i, 1] for i, t in enumerate([0, 0.5, 1, 3, 5, 10])}}
                    for curve in data['hazard_curves']
                ],
                style_table={'overflowX': 'auto'},
                page_size=10,
            ),
            
            html.H4("IR Curve", style={'margin-top': '20px'}),
            dash_table.DataTable(
                id='ir-curve-table',
                columns=[{"name": "Tenor (Years)", "id": "Tenor"}, {"name": "Rate", "id": "Rate"}],
                data=[{"Tenor": t, "Rate": r} for t, r in zip(data['ir_interpolator'].x, data['ir_interpolator'].y)],
                style_table={'overflowX': 'auto'},
                page_size=10,
            ),
            
            html.H4("Survival Probabilities", style={'margin-top': '20px'}),
            dash_table.DataTable(
                id='sp-table',
                columns=[{"name": "Date", "id": "Date"}, {"name": "Survival Probability", "id": "SP"}],
                data=[
                    {"Date": data['exposure_dates'][i].strftime('%Y-%m-%d'), "SP": data['sp'][i]}
                    for i in range(NO_OF_EXPOSURE_DATES)
                ],
                style_table={'overflowX': 'auto'},
                page_size=10,
            ),
        ]),
        
        # Results Tab
        dcc.Tab(label='Results', children=[
            html.H3("Computation Results", style={'margin-top': '20px'}),
            html.P("This tab stores the results of each computation. Select a result to view its details, graph, and data table."),
            dcc.Dropdown(
                id='result-selector',
                options=[],
                value=None,
                placeholder="Select a result to view",
                style={'margin-bottom': '20px'}
            ),
            # Summary Section
            html.Div(id='result-summary', children=[
                html.H4("Result Summary"),
                html.P(id='computation-type', children="Computation Type: N/A"),
                html.P(id='grid-id', children="Grid ID: N/A"),
                html.P(id='run-id', children="Run ID: N/A"),
                html.P(id='avg-capital', children="Average Capital: N/A"),
                html.P(id='max-capital', children="Maximum Capital: N/A"),
                html.P(id='max-capital-date', children="Date of Maximum Capital: N/A"),
            ], style={'padding': '10px', 'background-color': '#f0f0f0'}),
            dcc.Graph(id='result-graph'),
            dash_table.DataTable(
                id='result-table',
                columns=[{"name": "Date", "id": "Date"}, {"name": "Capital", "id": "Capital"}],
                data=[],
                style_table={'overflowX': 'auto'},
                page_size=10,
            ),
            # Interpretation Section
            html.Div([
                html.H4("Interpretation of Results"),
                html.P("Step 1: Understand the Capital Profile"),
                html.P("The graph and table show the capital requirement over 90 days. Each point represents the capital needed to cover CVA risk on that date. In a real application, this would be in a currency like dollars or euros."),
                html.P("Step 2: Analyze Trends"),
                html.P("Look for trends in the graph. An increasing trend might indicate rising risk exposure over time, while a decreasing trend might suggest risk mitigation. Spikes could indicate specific dates with high risk exposure."),
                html.P("Step 3: Compare Computations"),
                html.P("Compare EOD (entire portfolio) with Standalone (specific counterparty) to see the counterparty's contribution. Incremental Counterparty shows the additional capital if a new counterparty is added. Incremental Trade shows the impact of a new trade."),
                html.P("Step 4: Use Summary Metrics"),
                html.P("The summary above provides key metrics: average capital (overall risk level), maximum capital (peak risk), and the date of maximum capital (when peak risk occurs). Use these to assess the risk profile."),
                html.P("Step 5: Take Action"),
                html.P("If the maximum capital is too high, consider risk mitigation strategies (e.g., hedging, reducing exposure). If a new trade increases capital significantly, evaluate its cost-benefit. Use these insights to inform capital allocation and risk management decisions."),
            ], style={'padding': '10px', 'background-color': '#e0e0e0'}),
        ]),
    ]),
])

@app.callback(
    [Output("capital-profile", "figure"), Output("status", "children"),
     Output("result-selector", "options"), Output("result-store", "data")],
    [Input("btn-eod", "n_clicks"), Input("btn-standalone", "n_clicks"), 
     Input("btn-incremental-cpty", "n_clicks"), Input("btn-incremental-trade", "n_clicks")],
    [State("standalone-input", "value"), State("incremental-cpty-input", "value"), 
     State("incremental-trade-input", "value"), State("result-store", "data")],
    prevent_initial_call=True
)
def update_graph(n_eod, n_standalone, n_inc_cpty, n_inc_trade, standalone_id, inc_cpty_id, inc_trade_input, stored_results):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if stored_results is None:
        stored_results = []
    
    # Base capital profile for EOD
    base_profile = pd.DataFrame({
        't': [d.strftime('%Y-%m-%d') for d in data['exposure_dates']],  # Store as string
        'cap': np.random.normal(10000, 1000, NO_OF_EXPOSURE_DATES)  # Base portfolio capital
    })
    
    if button_id == "btn-eod":
        fig = px.line(base_profile, x='t', y='cap', title="EOD Capital Profile")
        result_entry = {
            'label': f"EOD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            'data': base_profile.to_dict('records'),
            'type': 'EOD',
            'grid_id': None,
            'run_id': None
        }
        stored_results.append(result_entry)
        options = [{'label': entry['label'], 'value': i} for i, entry in enumerate(stored_results)]
        return fig, "EOD computation completed.", options, stored_results
    
    elif button_id == "btn-standalone" and standalone_id:
        try:
            grid_id = int(standalone_id)
            if grid_id not in [1001, 1002, 1003, 1004, 1005]:
                return dash.no_update, "Error: Invalid Grid ID. Use 1001, 1002, 1003, 1004, or 1005.", dash.no_update, dash.no_update
            standalone_profile = base_profile.copy()
            standalone_profile['cap'] = standalone_profile['cap'] * 0.2 + grid_id * 10
            fig = px.line(standalone_profile, x='t', y='cap', title=f"Standalone Capital Profile for {grid_id}")
            result_entry = {
                'label': f"Standalone {grid_id} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                'data': standalone_profile.to_dict('records'),
                'type': 'Standalone',
                'grid_id': grid_id,
                'run_id': None
            }
            stored_results.append(result_entry)
            options = [{'label': entry['label'], 'value': i} for i, entry in enumerate(stored_results)]
            return fig, f"Standalone computation for {grid_id} completed.", options, stored_results
        except ValueError:
            return dash.no_update, "Error: Invalid Grid ID for Standalone.", dash.no_update, dash.no_update
    
    elif button_id == "btn-incremental-cpty" and inc_cpty_id:
        try:
            grid_id = int(inc_cpty_id)
            if grid_id not in [1001, 1002, 1003, 1004, 1005]:
                return dash.no_update, "Error: Invalid Grid ID. Use 1001, 1002, 1003, 1004, or 1005.", dash.no_update, dash.no_update
            inc_profile = base_profile.copy()
            inc_profile['cap'] = inc_profile['cap'] * 1.1
            fig = px.line(inc_profile, x='t', y='cap', title=f"Incremental Counterparty for {grid_id}")
            result_entry = {
                'label': f"Incremental Cpty {grid_id} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                'data': inc_profile.to_dict('records'),
                'type': 'Incremental Counterparty',
                'grid_id': grid_id,
                'run_id': None
            }
            stored_results.append(result_entry)
            options = [{'label': entry['label'], 'value': i} for i, entry in enumerate(stored_results)]
            return fig, f"Incremental Counterparty computation for {grid_id} completed.", options, stored_results
        except ValueError:
            return dash.no_update, "Error: Invalid Grid ID for Incremental Counterparty.", dash.no_update, dash.no_update
    
    elif button_id == "btn-incremental-trade" and inc_trade_input:
        try:
            grid_id, run_id = map(int, inc_trade_input.split(','))
            if grid_id not in [1001, 1002, 1003, 1004, 1005]:
                return dash.no_update, "Error: Invalid Grid ID. Use 1001, 1002, 1003, 1004, or 1005.", dash.no_update, dash.no_update
            trade_profile = base_profile.copy()
            trade_profile['cap'] = trade_profile['cap'] * 0.9
            fig = px.line(trade_profile, x='t', y='cap', title=f"Incremental Trade for Grid {grid_id}, Run {run_id}")
            result_entry = {
                'label': f"Incremental Trade {grid_id},{run_id} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                'data': trade_profile.to_dict('records'),
                'type': 'Incremental Trade',
                'grid_id': grid_id,
                'run_id': run_id
            }
            stored_results.append(result_entry)
            options = [{'label': entry['label'], 'value': i} for i, entry in enumerate(stored_results)]
            return fig, f"Incremental Trade computation for Grid {grid_id}, Run {run_id} completed.", options, stored_results
        except (ValueError, IndexError):
            return dash.no_update, "Error: Invalid input for Incremental Trade. Use format: Grid ID,Run ID (e.g., 1001,123)", dash.no_update, dash.no_update
    
    return dash.no_update, "Please provide valid input and click a button.", dash.no_update, dash.no_update

@app.callback(
    [Output("result-graph", "figure"), Output("result-table", "data"),
     Output("computation-type", "children"), Output("grid-id", "children"),
     Output("run-id", "children"), Output("avg-capital", "children"),
     Output("max-capital", "children"), Output("max-capital-date", "children")],
    [Input("result-selector", "value")],
    [State("result-store", "data")]
)
def update_result_view(selected_result, stored_results):
    if selected_result is None or stored_results is None or not stored_results:
        return px.line(), [], "Computation Type: N/A", "Grid ID: N/A", "Run ID: N/A", "Average Capital: N/A", "Maximum Capital: N/A", "Date of Maximum Capital: N/A"
    
    result_data = pd.DataFrame(stored_results[selected_result]['data'])
    fig = px.line(result_data, x='t', y='cap', title=stored_results[selected_result]['label'])
    table_data = [
        {"Date": row['t'], "Capital": row['cap']}
        for _, row in result_data.iterrows()
    ]
    
    # Summary Metrics
    avg_capital = result_data['cap'].mean()
    max_capital = result_data['cap'].max()
    max_capital_idx = result_data['cap'].idxmax()
    max_capital_date = result_data.loc[max_capital_idx, 't']
    
    # Additional Details
    computation_type = stored_results[selected_result]['type']
    grid_id = stored_results[selected_result]['grid_id']
    run_id = stored_results[selected_result]['run_id']
    
    return (fig, table_data,
            f"Computation Type: {computation_type}",
            f"Grid ID: {grid_id if grid_id is not None else 'N/A'}",
            f"Run ID: {run_id if run_id is not None else 'N/A'}",
            f"Average Capital: {avg_capital:.2f}",
            f"Maximum Capital: {max_capital:.2f}",
            f"Date of Maximum Capital: {max_capital_date}")

if __name__ == '__main__':
    app.run(debug=True)