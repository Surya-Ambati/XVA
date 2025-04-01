import numpy as np
from scipy.optimize import root_scalar

def calibrate_hazard_rates(cds_spreads, maturities, recovery=0.4, risk_free_rate=0.02):
    """
    Bootstraps piecewise constant hazard rates from CDS spreads.
    """
    # Convert spreads from bps to decimal
    cds_spreads = [s / 10000 for s in cds_spreads]  # bps to decimal
    
    hazard_rates = []
    survival_probs = [1.0]  # P(τ > 0) = 1
    
    for i, (T, spread) in enumerate(zip(maturities, cds_spreads)):
        prev_T = maturities[i-1] if i > 0 else 0.0
        dt = T - prev_T
        
        def equation(lambda_i):
            # Premium leg: CDS spread × time × survival probability × discount factor
            premium_leg = spread * dt * survival_probs[-1] * np.exp(-risk_free_rate * T)
            
            # Protection leg: (1-R) × hazard × time × survival × discount
            # Adjusted survival probability with current lambda
            avg_survival = survival_probs[-1] * np.exp(-lambda_i * dt/2)  # Midpoint approximation
            protection_leg = (1 - recovery) * lambda_i * dt * avg_survival * np.exp(-risk_free_rate * T)
            
            return premium_leg - protection_leg
        
        # Wider bracket and try-except for robustness
        try:
            sol = root_scalar(equation, bracket=[0.0001, 1.0], method='brentq')
            hazard_rate = sol.root
        except ValueError:
            # If bracket fails, try a different method
            sol = root_scalar(equation, x0=0.01, method='newton')
            hazard_rate = sol.root
            
        hazard_rates.append(hazard_rate)
        # Calculate survival probability using the full time from 0 to T
        total_hazard = sum(h * (maturities[j] - (maturities[j-1] if j > 0 else 0)) 
                         for j, h in enumerate(hazard_rates))
        survival_probs.append(np.exp(-total_hazard))
    
    return hazard_rates, survival_probs[1:]

import dash
from dash import dcc, html
import plotly.graph_objects as go

# Sample data
maturities = [1, 3, 5, 7, 10]
cds_spreads = [100, 150, 200, 220, 250]  # in bps
hazard_rates, survival_probs = calibrate_hazard_rates(cds_spreads, maturities)

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("CDS Hazard Rate & Survival Probability Curve"),
    dcc.Graph(
        figure={
            'data': [
                go.Scatter(x=maturities, y=[h*100 for h in hazard_rates],  # Convert to percentage
                         name='Hazard Rate (%)', mode='lines+markers'),
                go.Scatter(x=maturities, y=survival_probs, 
                         name='Survival Probability', yaxis='y2', mode='lines+markers')
            ],
            'layout': {
                'yaxis': {'title': 'Hazard Rate (%)'},
                'yaxis2': {'title': 'Survival Probability', 'overlaying': 'y', 'side': 'right'},
                'xaxis': {'title': 'Maturity (Years)'}
            }
        }
    )
])

if __name__ == '__main__':
    app.run(debug=True)