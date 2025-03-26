import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

# Function to calculate Bilateral CVA (BCVA)
def calculate_bilateral_cva(pos_exposure, neg_exposure, hazard_rate_c_pct, hazard_rate_b_pct, 
                           recovery_rate_c_pct, recovery_rate_b_pct, discount_rate_pct, time_steps):
    """
    Calculate Bilateral CVA (BCVA) using the expectation method, including CVA and DVA terms.

    Parameters:
    - pos_exposure (list): Positive exposure (V_t)^+ at each time step after t=0.
    - neg_exposure (list): Negative exposure (V_t)^- at each time step after t=0 (absolute values).
    - hazard_rate_c_pct (float): Counterparty hazard rate (λ_C, %).
    - hazard_rate_b_pct (float): Bank hazard rate (λ_B, %).
    - recovery_rate_c_pct (float): Counterparty recovery rate (R_C, %).
    - recovery_rate_b_pct (float): Bank recovery rate (R_B, %).
    - discount_rate_pct (float): Risk-free rate (r, %).
    - time_steps (list): Time points (e.g., [0, 0.5, 1, ...]).

    Returns:
    - bcva (float): Bilateral CVA (CVA + DVA).
    - cva (float): Counterparty CVA (cost).
    - dva (float): Bank DVA (benefit, negative).
    - survival_c (list): Counterparty survival probabilities.
    - survival_b (list): Bank survival probabilities.
    - default_probs_c_pct (list): Counterparty default probabilities (%).
    - default_probs_b_pct (list): Bank default probabilities (%).
    - discounted_pos_exposures (list): Discounted positive exposures.
    - discounted_neg_exposures (list): Discounted negative exposures.

    Steps:
    1. Convert inputs to decimals (e.g., λ_C = hazard_rate_c_pct / 100).
    2. Compute survival probabilities: Φ(τ_C > t) = e^(-λ_C t), Φ(τ_B > t) = e^(-λ_B t).
    3. Calculate default probabilities: [Φ(τ > t_i) - Φ(τ > t_{i+1})].
    4. Apply discount factors: e^(-r t).
    5. CVA = (1 - R_C) Σ [Φ(τ_C > t_i) - Φ(τ_C > t_{i+1})] Φ(τ_B > t_{i+1}) e^(-r t) (V_t)^+.
    6. DVA = (1 - R_B) Σ [Φ(τ_B > t_i) - Φ(τ_B > t_{i+1})] Φ(τ_C > t_{i+1}) e^(-r t) (V_t)^- (negative).
    7. BCVA = CVA + DVA.
    """
    # Step 1: Convert percentages to decimals
    lambda_c = hazard_rate_c_pct / 100  # λ_C
    lambda_b = hazard_rate_b_pct / 100  # λ_B
    r_c = recovery_rate_c_pct / 100  # R_C
    r_b = recovery_rate_b_pct / 100  # R_B
    r = discount_rate_pct / 100  # r
    
    n = len(time_steps) - 1  # Number of intervals

    # Step 2: Compute survival probabilities
    survival_c = [np.exp(-lambda_c * t) for t in time_steps]  # Φ(τ_C > t)
    survival_b = [np.exp(-lambda_b * t) for t in time_steps]  # Φ(τ_B > t)

    # Step 3: Calculate default probabilities
    default_probs_c = [survival_c[i] - survival_c[i + 1] for i in range(n)]  # Φ(τ_C > t_i) - Φ(τ_C > t_{i+1})
    default_probs_b = [survival_b[i] - survival_b[i + 1] for i in range(n)]  # Φ(τ_B > t_i) - Φ(τ_B > t_{i+1})
    default_probs_c_pct = [p * 100 for p in default_probs_c]
    default_probs_b_pct = [p * 100 for p in default_probs_b]

    # Step 4: Apply discount factors
    discount_factors = [np.exp(-r * t) for t in time_steps[1:]]  # e^(-r t)
    discounted_pos_exposures = [pos_exposure[i] * discount_factors[i] for i in range(n)]
    discounted_neg_exposures = [neg_exposure[i] * discount_factors[i] for i in range(n)]

    # Step 5: Calculate CVA
    cva = (1 - r_c) * sum(
        default_probs_c[i] * survival_b[i + 1] * discounted_pos_exposures[i]
        for i in range(n)
    )

    # Step 6: Calculate DVA (negative exposure makes this term negative)
    dva = (1 - r_b) * sum(
        default_probs_b[i] * survival_c[i + 1] * discounted_neg_exposures[i]
        for i in range(n)
    )

    # Step 7: Bilateral CVA
    bcva = cva + dva  # DVA is negative, reducing the net adjustment

    return bcva, cva, dva, survival_c, survival_b, default_probs_c_pct, default_probs_b_pct, discounted_pos_exposures, discounted_neg_exposures

# Dash App
app = Dash(__name__)

# Example data (hypothetical, as Table 3.3 isn’t provided)
time_steps = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
pos_exposure = [700000, 850000, 900000, 870000, 800000, 730000, 600000, 470000, 290000, 90000]  # (V_t)^+
neg_exposure = [50000, 60000, 70000, 65000, 60000, 55000, 50000, 40000, 30000, 20000]  # |(V_t)^-|

# Layout
app.layout = html.Div([
    html.H1("Bilateral CVA (CVA + DVA)", style={'margin-bottom': '5px'}),
    
    # Two-column layout
    html.Div([
        # Left Column: Inputs and Notes
        html.Div([
            # Inputs
            html.Div([
                html.Label("Counterparty Hazard Rate (λ_C, %):"),
                dcc.Slider(id='hazard-c', min=1, max=10, step=0.5, value=3, marks={1: '1%', 10: '10%'}),
                
                html.Label("Bank Hazard Rate (λ_B, %):"),
                dcc.Slider(id='hazard-b', min=1, max=10, step=0.5, value=2, marks={1: '1%', 10: '10%'}),
                
                html.Label("Counterparty Recovery Rate (R_C, %):"),
                dcc.Slider(id='recovery-c', min=0, max=100, step=5, value=40, marks={0: '0%', 100: '100%'}),
                
                html.Label("Bank Recovery Rate (R_B, %):"),
                dcc.Slider(id='recovery-b', min=0, max=100, step=5, value=40, marks={0: '0%', 100: '100%'}),
                
                html.Label("Discount Rate (r, %):"),
                dcc.Slider(id='discount-rate', min=1, max=10, step=0.5, value=3, marks={1: '1%', 10: '10%'}),
                
                html.Label("Positive Exposure at t=0.5 ($):"), dcc.Input(id='pos-t05', type='number', value=700000),
                html.Label("Negative Exposure at t=0.5 ($):"), dcc.Input(id='neg-t05', type='number', value=50000),
                # Add more inputs for simplicity (limiting to t=0.5 here; extend as needed)
            ], style={'padding': '5px'}),
            
            # Notes
            html.Div([
                html.H2("Notes on Bilateral CVA", style={'margin-top': '10px'}),
                html.Ul([
                    html.Li("Bilateral CVA = CVA + DVA: Accounts for both counterparty (C) and bank (B) defaults."),
                    html.Li("CVA: Cost if C defaults first (τ_C < τ_B < T), (1 - R_C) times positive exposure (V_t)^+."),
                    html.Li("DVA: Benefit if B defaults first (τ_B < τ_C < T), (1 - R_B) times negative exposure (V_t)^-, negative term."),
                    html.Li("Formula: BCVA(t) = ∫ λ_C e^(-∫(λ_C + λ_B)) E[(1 - R_C) (V_s)^+] ds + ∫ λ_B e^(-∫(λ_C + λ_B)) E[(1 - R_B) (V_s)^-] ds."),
                    html.Li("Discrete: (1 - R_C) Σ [Φ(τ_C > t_i) - Φ(τ_C > t_{i+1})] Φ(τ_B > t_{i+1}) e^(-r t) (V_t)^+ + (1 - R_B) Σ [Φ(τ_B > t_i) - Φ(τ_B > t_{i+1})] Φ(τ_C > t_{i+1}) e^(-r t) (V_t)^-."),
                    html.Li("Assumption: Independent defaults (τ_C and τ_B), no simultaneous defaults.")
                ])
            ], style={'padding': '5px'})
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        # Right Column: Graphs and Outputs
        html.Div([
            dcc.Graph(id='survival-graph', style={'height': '400px'}),
            dcc.Graph(id='exposure-graph', style={'height': '400px'}),
            dcc.Graph(id='default-prob-graph', style={'height': '400px'}),
            html.H3(id='bcva-value', style={'margin-top': '5px'}),
            html.P(id='interpretation', style={'margin-top': '5px'}),
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '5px'}),
    ], style={'display': 'flex', 'flex-direction': 'row'})
])

# Callback to update graphs and outputs
@app.callback(
    [Output('survival-graph', 'figure'), Output('exposure-graph', 'figure'), 
     Output('default-prob-graph', 'figure'), Output('bcva-value', 'children'), 
     Output('interpretation', 'children')],
    [Input('hazard-c', 'value'), Input('hazard-b', 'value'), Input('recovery-c', 'value'),
     Input('recovery-b', 'value'), Input('discount-rate', 'value'), Input('pos-t05', 'value'),
     Input('neg-t05', 'value')]
)
def update_graph(hazard_c, hazard_b, recovery_c, recovery_b, discount_rate, pos_t05, neg_t05):
    # Example data extended for simplicity
    pos_exposure = [pos_t05] + [850000] * (len(time_steps) - 2)  # Placeholder beyond t=0.5
    neg_exposure = [neg_t05] + [60000] * (len(time_steps) - 2)
    
    bcva, cva, dva, survival_c, survival_b, default_probs_c_pct, default_probs_b_pct, discounted_pos_exposures, discounted_neg_exposures = calculate_bilateral_cva(
        pos_exposure, neg_exposure, hazard_c, hazard_b, recovery_c, recovery_b, discount_rate, time_steps
    )
    
    # Graph 1: Survival Probabilities
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=time_steps, y=survival_c, name='Counterparty Survival (Φ_C)'))
    fig1.add_trace(go.Scatter(x=time_steps, y=survival_b, name='Bank Survival (Φ_B)'))
    fig1.update_layout(
        title='Survival Probabilities Over Time',
        xaxis_title='Time (Years)',
        yaxis_title='Survival Probability',
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    # Graph 2: Discounted Exposures
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time_steps[1:], y=discounted_pos_exposures, name='Discounted Positive Exposure'))
    fig2.add_trace(go.Scatter(x=time_steps[1:], y=discounted_neg_exposures, name='Discounted Negative Exposure'))
    fig2.update_layout(
        title='Discounted Exposures Over Time',
        xaxis_title='Time (Years)',
        yaxis_title='Exposure ($)',
        height=400
    )
    
    # Graph 3: Default Probabilities
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=time_steps[1:], y=default_probs_c_pct, name='Counterparty Default Prob (%)'))
    fig3.add_trace(go.Bar(x=time_steps[1:], y=default_probs_b_pct, name='Bank Default Prob (%)'))
    fig3.update_layout(
        title='Default Probabilities Over Time',
        xaxis_title='Time (Years)',
        yaxis_title='Default Probability (%)',
        height=400
    )
    
    # Outputs
    bcva_value = f"Bilateral CVA: ${bcva:,.2f} (CVA: ${cva:,.2f}, DVA: ${dva:,.2f})"
    interpretation = (
        f"Interpretation: CVA (${cva:,.2f}) is the cost if the counterparty defaults first. "
        f"DVA (${dva:,.2f}) is the benefit if the bank defaults first. Net BCVA is ${bcva:,.2f}."
    )
    
    return fig1, fig2, fig3, bcva_value, interpretation

# Run the app
if __name__ == '__main__':
    app.run(debug=True)