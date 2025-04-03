import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

# Dummy Data Generation
np.random.seed(42)
n_paths = 1000  # Monte Carlo paths
n_steps = 10    # Time steps (e.g., years)
time = np.linspace(0, 10, n_steps)
discount_rate = 0.03  # Constant risk-free rate
recovery_rate = 0.4   # 40% recovery
hazard_rate = 0.02    # Flat 2% hazard rate

# Simulate exposure paths (e.g., interest rate swap-like)
exposure = np.abs(np.random.normal(100, 50, (n_paths, n_steps)))  # Positive exposures
exposure = np.maximum(exposure, 0)  # Ensure positive (V^+)

# Survival probabilities
survival_prob = np.exp(-hazard_rate * time)
default_prob = np.diff(1 - survival_prob, prepend=1)  # Probability of default in each interval

# Function to calculate CVA with WWR using Gaussian Copula
def calculate_cva_wwr(exposure, default_prob, discount_rate, recovery_rate, rho):
    n_paths, n_steps = exposure.shape
    cva = np.zeros(n_steps)
    
    for t in range(n_steps):
        # Empirical CDF of exposure at time t
        sorted_exposure = np.sort(exposure[:, t])
        cdf_exposure = (np.arange(1, n_paths + 1) - 0.5) / n_paths  # Shifted to avoid 0 or 1
        
        # Convert to standard normal
        u = norm.ppf(cdf_exposure)
        v = norm.ppf(survival_prob[t]) if t < n_steps - 1 else -np.inf  # Handle boundary
        
        # Bivariate Gaussian Copula
        cov = [[1, rho], [rho, 1]]  # Covariance matrix
        biv_norm = multivariate_normal(mean=[0, 0], cov=cov, allow_singular=True)  # Allow singular
        
        # Calculate conditional probability for each exposure level
        cond_probs = []
        for i in range(n_paths):
            p_j = sorted_exposure[i]
            u_j = u[i]
            if i == 0:
                p_j_prev = 0
                u_j_prev = -np.inf
            else:
                p_j_prev = sorted_exposure[i-1]
                u_j_prev = u[i-1]
            
            # Equation 7.9: C2 terms
            term1 = biv_norm.cdf([u_j, v])
            term2 = biv_norm.cdf([u_j_prev, v])
            term3 = biv_norm.cdf([u_j, norm.ppf(survival_prob[t-1])]) if t > 0 else 0
            term4 = biv_norm.cdf([u_j_prev, norm.ppf(survival_prob[t-1])]) if t > 0 else 0
            prob = (term1 - term2 - term3 + term4) * default_prob[t]
            cond_probs.append(max(prob, 0))  # Ensure non-negative
        
        # Normalize probabilities
        cond_probs = np.array(cond_probs)
        cond_probs = cond_probs / np.sum(cond_probs) if np.sum(cond_probs) > 0 else np.ones(n_paths) / n_paths
        
        # Weighted exposure for CVA
        weighted_exposure = np.sum(sorted_exposure * cond_probs)
        discount_factor = np.exp(-discount_rate * time[t])
        cva[t] = (1 - recovery_rate) * discount_factor * weighted_exposure
    
    return np.sum(cva)

# Calculate CVA for different correlations
rhos = np.linspace(-1, 1, 21)  # From -1 (RWR) to 1 (WWR)
cva_values = [calculate_cva_wwr(exposure, default_prob, discount_rate, recovery_rate, rho) for rho in rhos]

# Dash App
app = Dash(__name__)

app.layout = html.Div([
    html.H1("CVA with Wrong-Way and Right-Way Risk"),
    dcc.Graph(id='cva-graph'),
    dcc.Slider(
        id='rho-slider',
        min=-1,
        max=1,
        step=0.1,
        value=0,
        marks={i: f'{i:.1f}' for i in np.linspace(-1, 1, 11)}
    ),
    html.Div(id='cva-output')
])

@app.callback(
    [Output('cva-graph', 'figure'), Output('cva-output', 'children')],
    [Input('rho-slider', 'value')]
)
def update_graph(selected_rho):
    # Full plot data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rhos, y=cva_values, mode='lines+markers', name='CVA vs Rho'))
    fig.add_trace(go.Scatter(x=[selected_rho], y=[calculate_cva_wwr(exposure, default_prob, discount_rate, recovery_rate, selected_rho)],
                             mode='markers', marker=dict(size=15, color='red'), name='Selected Rho'))
    fig.update_layout(title='CVA as a Function of Correlation (Rho)',
                      xaxis_title='Correlation (Rho)', yaxis_title='CVA (£)')
    
    # Selected CVA value
    cva_selected = calculate_cva_wwr(exposure, default_prob, discount_rate, recovery_rate, selected_rho)
    output_text = f"CVA at Rho = {selected_rho:.1f}: £{cva_selected:,.2f}"
    
    return fig, output_text

# User Example
print("User Example:")
print("Suppose you're a bank trading a 10-year deal with a counterparty.")
print("With no correlation (Rho = 0), CVA might be £4,000.")
print("Slide Rho to 1 (WWR), and CVA could rise to £5,500 due to higher exposure at default.")
print("Slide Rho to -1 (RWR), and CVA might drop to £2,500 as exposure falls when default risk rises.")

if __name__ == '__main__':
    app.run(debug=True)