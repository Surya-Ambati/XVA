import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

# Function to calculate Unilateral CVA
def calculate_unilateral_cva(exposure, hazard_rate_pct, recovery_rate_pct, discount_rate_pct, time_steps):
    """
    Calculate Unilateral CVA using the discrete summation formula.
    
    Parameters:
    - exposure (list): Positive exposure (V_t)^+ at each time step.
    - hazard_rate_pct (float): Annual default probability as percentage (e.g., 5 for 5%).
    - recovery_rate_pct (float): Recovery rate as percentage (e.g., 40 for 40%).
    - discount_rate_pct (float): Annual risk-free rate as percentage (e.g., 3 for 3%).
    - time_steps (list): Time points (e.g., [0, 1, 2] years).
    
    Returns:
    - cva (float): Credit Valuation Adjustment.
    - survival_probs (list): Survival probabilities at each time step.
    - default_probs_pct (list): Default probabilities between time steps as percentages.
    - discounted_exposures (list): Discounted exposures at each time step.
    """
    hazard_rate = hazard_rate_pct / 100
    recovery_rate = recovery_rate_pct / 100
    discount_rate = discount_rate_pct / 100
    
    n = len(time_steps) - 1
    survival_probs = [np.exp(-hazard_rate * t) for t in time_steps]
    default_probs = [survival_probs[i] - survival_probs[i + 1] for i in range(n)]
    default_probs_pct = [prob * 100 for prob in default_probs]
    discount_factors = [np.exp(-discount_rate * t) for t in time_steps[1:]]
    discounted_exposures = [exposure[i] * discount_factors[i] for i in range(n)]
    
    cva = (1 - recovery_rate) * sum(default_prob * discounted_exp 
                                    for default_prob, discounted_exp in zip(default_probs, discounted_exposures))
    
    return cva, survival_probs, default_probs_pct, discounted_exposures

# Dash App
app = Dash(__name__)

# Example data
time_steps = [0, 1, 2]
exposure = [100, 80]
hazard_rate_pct = 5
recovery_rate_pct = 40
discount_rate_pct = 3

# Layout
app.layout = html.Div([
    html.H1("Unilateral CVA Calculator and Explanation", style={'margin-bottom': '5px'}),
    
    # Two-column layout
    html.Div([
        # Left Column: Inputs and Notes
        html.Div([
            # Inputs
            html.Div([
                html.Label("Hazard Rate (λ_C, annual default probability, %):"),
                dcc.Slider(id='hazard-rate', min=1, max=100, step=1, value=5, marks={1: '1%', 100: '100%'}),
                
                html.Label("Recovery Rate (R, %):"),
                dcc.Slider(id='recovery-rate', min=0, max=100, step=5, value=40, marks={0: '0%', 100: '100%'}),
                
                html.Label("Discount Rate (r, %):"),
                dcc.Slider(id='discount-rate', min=1, max=100, step=1, value=0, marks={1: '1%', 100: '100%'}),
                
                html.Label("Exposure at t=1 ($):"),
                dcc.Input(id='exposure-t1', type='number', value=100),
                
                html.Label("Exposure at t=2 ($):"),
                dcc.Input(id='exposure-t2', type='number', value=80),
            ], style={'padding': '5px'}),
            
            # Notes (directly below inputs)
            html.Div([
                html.H2("CVA Formula Breakdown", style={'margin-top': '10px'}),
                html.P("The Unilateral CVA formula (discrete version) is:"),
                html.P("CVA(t) = (1 - R) × Σ [Φ(τ > t_i) - Φ(τ > t_{i+1})] × E[e^(-∫ r(u) du) (V_{t_i})^+ | G_t]"),
                html.Ul([
                    html.Li("1 - R: Loss given default (1 - recovery rate). If R = 40%, then 1 - R = 60%."),
                    html.Li("Φ(τ > t_i): Survival probability to time t_i, calculated as e^(-λ_C * t_i)."),
                    html.Li("Φ(τ > t_i) - Φ(τ > t_{i+1}): Default probability between t_i and t_{i+1}, in percentage."),
                    html.Li("e^(-∫ r(u) du): Discount factor from t_i to today, approximated as e^(-r * t_i)."),
                    html.Li("(V_{t_i})^+: Positive exposure at time t_i (what counterparty owes, in $)."),
                    html.Li("Sum over i: Adds up the expected loss for each time interval.")
                ]),
                html.H2("Step-by-Step Example", style={'margin-top': '10px'}),
                html.P("Using t = [0, 1, 2], λ_C = 5%, R = 40%, r = 3%, V_1 = 100, V_2 = 80:"),
                html.Ul([
                    html.Li("Survival: Φ(τ > 1) = e^(-0.05 * 1) = 0.951, Φ(τ > 2) = e^(-0.05 * 2) = 0.905."),
                    html.Li("Default probs: [0.951 - 0.905 = 4.6%, 0.905 - 0 = 90.5%]."),
                    html.Li("Discount: e^(-0.03 * 1) = 0.970, e^(-0.03 * 2) = 0.942."),
                    html.Li("Discounted exposure: [100 * 0.970 = 97, 80 * 0.942 = 75.36]."),
                    html.Li("CVA = 0.6 * [0.046 * 97 + 0.905 * 75.36] = 0.6 * 72.663 = 43.60.")
                ]),
                html.H2("Notes from the Book", style={'margin-top': '10px'}),
                html.P("CVA adjusts for counterparty default risk. Key points:"),
                html.Ul([
                    html.Li("Calculated at counterparty level due to netting rules."),
                    html.Li("Managed separately from derivatives, requires credit risk skills."),
                    html.Li("Reported separately in financial statements."),
                    html.Li("Unilateral CVA assumes only counterparty defaults; bilateral includes DVA.")
                ])
            ], style={'padding': '5px'})
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        # Right Column: Graphs and CVA value
        html.Div([
            dcc.Graph(id='survival-prob-graph', style={'height': '400px'}),
            dcc.Graph(id='discounted-exp-graph', style={'height': '400px'}),
            dcc.Graph(id='default-prob-graph', style={'height': '400px'}),
            html.H3(id='cva-value', style={'margin-top': '5px'}),
            html.P(id='cva-interpretation', style={'margin-top': '5px'}),
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '5px'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}) # this is a flexbox layout helps to keep content in right side
])

# Callback to update graphs and CVA value
@app.callback(
    [Output('survival-prob-graph', 'figure'), Output('discounted-exp-graph', 'figure'), 
     Output('default-prob-graph', 'figure'), Output('cva-value', 'children'), 
     Output('cva-interpretation', 'children')],
    [Input('hazard-rate', 'value'), Input('recovery-rate', 'value'), 
     Input('discount-rate', 'value'), Input('exposure-t1', 'value'), 
     Input('exposure-t2', 'value')]
)
def update_graph(hazard_rate_pct, recovery_rate_pct, discount_rate_pct, exp_t1, exp_t2):
    exposure = [exp_t1, exp_t2]
    cva, survival_probs, default_probs_pct, discounted_exposures = calculate_unilateral_cva(
        exposure, hazard_rate_pct, recovery_rate_pct, discount_rate_pct, time_steps
    )
    
    # Graph 1: Survival Probability
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=time_steps, y=[1] + survival_probs[1:], name='Survival Probability'))
    fig1.update_layout(
        title='Survival Probability Over Time',
        xaxis_title='Time (Years)',
        yaxis_title='Survival Probability',
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    # Graph 2: Discounted Exposure
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time_steps[1:], y=discounted_exposures, name='Discounted Exposure ($)'))
    fig2.update_layout(
        title='Discounted Exposure Over Time',
        xaxis_title='Time (Years)',
        yaxis_title='Discounted Exposure ($)',
        height=400
    )
    
    # Graph 3: Default Probability
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=time_steps[1:], y=default_probs_pct, name='Default Probability (%)'))
    fig3.update_layout(
        title='Default Probability Over Time',
        xaxis_title='Time (Years)',
        yaxis_title='Default Probability (%)',
        yaxis=dict(range=[0, 100]),
        height=400
    )
    
    # CVA Interpretation
    original_value = sum(exposure)
    economic_value = original_value - cva
    interpretation = (
        f"Interpretation: The original portfolio value (undiscounted sum of exposures) is ${original_value:.2f}. "
        f"With a CVA of ${cva:.2f}, the economic value (adjusted for credit risk) is ${economic_value:.2f}."
    )
    
    return fig1, fig2, fig3, f"Calculated CVA: ${cva:.2f}", interpretation

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)