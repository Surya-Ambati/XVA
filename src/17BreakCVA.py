import numpy as np

def calculate_cva_netting(trade_values, default_probs, recovery_rate):
    """
    Calculate unilateral CVA with and without close-out netting.
    
    Parameters:
    - trade_values: List of trade values (positive = they owe you, negative = you owe them)
    - default_probs: List of default probabilities at each time step
    - recovery_rate: Recovery rate (e.g., 0.4 means recover 40%)
    
    Returns:
    - cva_non_netted: CVA without netting
    - cva_netted: CVA with netting
    """
    # Without netting: Sum exposures for each trade separately
    exposures_non_netted = [max(v, 0) for v in trade_values]
    cva_non_netted = sum((1 - recovery_rate) * prob * exp 
                         for prob, exp in zip(default_probs, exposures_non_netted))
    
    # With netting: Net the trade values first, then take exposure
    net_value = sum(trade_values)
    exposure_netted = max(net_value, 0)
    cva_netted = (1 - recovery_rate) * sum(prob for prob in default_probs) * exposure_netted
    
    return cva_non_netted, cva_netted

# Example
trade_values = [100, -60, 20]  # Three trades
default_probs = [0.05]  # Simplified: 5% default probability
recovery_rate = 0.4

cva_non_netted, cva_netted = calculate_cva_netting(trade_values, default_probs, recovery_rate)
print(f"CVA without netting: ${cva_non_netted:.2f}")
print(f"CVA with netting: ${cva_netted:.2f}")


def calculate_cva_break_clause(exposures, default_probs, recovery_rate, break_date):
    """
    Calculate CVA with a mandatory break clause.
    
    Parameters:
    - exposures: List of exposures over time
    - default_probs: List of default probabilities at each time step
    - recovery_rate: Recovery rate
    - break_date: Time step after which the contract ends (exposure = 0)
    
    Returns:
    - cva_no_break: CVA without break
    - cva_with_break: CVA with break
    """
    # Without break: Use full exposure profile
    cva_no_break = sum((1 - recovery_rate) * prob * exp 
                       for prob, exp in zip(default_probs, exposures))
    
    # With break: Set exposures to 0 after break_date
    exposures_with_break = exposures.copy()
    for i in range(break_date, len(exposures)):
        exposures_with_break[i] = 0
    cva_with_break = sum((1 - recovery_rate) * prob * exp 
                         for prob, exp in zip(default_probs, exposures_with_break))
    
    return cva_no_break, cva_with_break

# Example
exposures = [10000, 10000, 10000]  # Exposure over 3 years
default_probs = [0.02, 0.03, 0.04]  # Default probabilities
recovery_rate = 0.4
break_date = 2  # Break after year 2

cva_no_break, cva_with_break = calculate_cva_break_clause(exposures, default_probs, recovery_rate, break_date)
print(f"CVA without break: ${cva_no_break:.2f}")
print(f"CVA with break: ${cva_with_break:.2f}")


def calculate_cva_csa_simple(exposures, default_probs, recovery_rate, threshold):
    """
    Calculate CVA with a CSA using the simple model (threshold).
    
    Parameters:
    - exposures: List of exposures over time
    - default_probs: List of default probabilities
    - recovery_rate: Recovery rate
    - threshold: CSA threshold (caps exposure)
    
    Returns:
    - cva_no_csa: CVA without CSA
    - cva_with_csa: CVA with CSA
    """
    # Without CSA
    cva_no_csa = sum((1 - recovery_rate) * prob * exp 
                     for prob, exp in zip(default_probs, exposures))
    
    # With CSA: Cap exposure at the threshold
    exposures_with_csa = [min(max(exp, 0), threshold) for exp in exposures]
    print(f"Exposures with CSA: {exposures_with_csa}")
    cva_with_csa = sum((1 - recovery_rate) * prob * exp 
                       for prob, exp in zip(default_probs, exposures_with_csa))
    
    return cva_no_csa, cva_with_csa

# Example
exposures = [10000, 12000, 8000]  # Exposure over 3 years
default_probs = [0.02, 0.03, 0.04]
recovery_rate = 0.4
threshold = 3000  # CSA threshold

cva_no_csa, cva_with_csa = calculate_cva_csa_simple(exposures, default_probs, recovery_rate, threshold)
print(f"CVA without CSA: ${cva_no_csa:.2f}")
print(f"CVA with CSA: ${cva_with_csa:.2f}")



def calculate_cva_downgrade_trigger(exposures, default_probs, recovery_rate, thresholds_by_rating, transition_probs):
    """
    Calculate CVA with downgrade triggers using weighted expected exposure.
    
    Parameters:
    - exposures: List of exposures
    - default_probs: List of default probabilities
    - recovery_rate: Recovery rate
    - thresholds_by_rating: Dict of thresholds for each rating (e.g., {'A': 5000, 'BBB': 2000})
    - transition_probs: Dict of probabilities to transition to each rating
    
    Returns:
    - cva: Weighted CVA
    """
    ratings = list(thresholds_by_rating.keys())
    weighted_exposures = [0] * len(exposures)
    
    for i, exp in enumerate(exposures):
        for rating in ratings:
            # Cap exposure at the threshold for this rating
            capped_exp = min(max(exp, 0), thresholds_by_rating[rating])
            weighted_exposures[i] += transition_probs[rating] * capped_exp
    
    cva = sum((1 - recovery_rate) * prob * exp 
              for prob, exp in zip(default_probs, weighted_exposures))
    
    return cva

# Example
thresholds_by_rating = {'A': 5000, 'BBB': 2000, 'BB': 0}
transition_probs = {'A': 0.7, 'BBB': 0.2, 'BB': 0.1}

cva_downgrade = calculate_cva_downgrade_trigger(exposures, default_probs, recovery_rate, thresholds_by_rating, transition_probs)
print(f"CVA with downgrade trigger: ${cva_downgrade:.2f}")


def calculate_cva_default_waterfall(derivative_values, loan_principal, security_values, default_probs):
    """
    Calculate CVA with non-financial security for Case 1 (junior) and Case 2 (pari passu).
    
    Parameters:
    - derivative_values: List of derivative values over time
    - loan_principal: Loan principal (constant)
    - security_values: List of security values over time
    - default_probs: List of default probabilities
    
    Returns:
    - cva_case1: CVA for Case 1 (junior)
    - cva_case2: CVA for Case 2 (pari passu)
    """
    cva_case1 = 0
    cva_case2 = 0
    
    for i in range(len(derivative_values)):
        V = derivative_values[i]
        S = security_values[i]
        prob = default_probs[i]
        
        # Case 1: Derivative ranks junior
        remaining_after_loan = max(S - loan_principal, 0)
        loss_case1 = max(V - remaining_after_loan, 0)
        cva_case1 += prob * loss_case1
        
        # Case 2: Derivative ranks pari passu
        total_claim = loan_principal + V
        total_loss = max(total_claim - S, 0)
        loss_fraction = total_loss / total_claim if total_claim > 0 else 0
        loss_case2 = V * loss_fraction
        cva_case2 += prob * loss_case2
    
    return cva_case1, cva_case2

# Example
derivative_values = [20000, 20000, 20000]
loan_principal = 100000
security_values = [80000, 80000, 80000]
default_probs = [0.01, 0.01, 0.01]

cva_case1, cva_case2 = calculate_cva_default_waterfall(derivative_values, loan_principal, security_values, default_probs)
print(f"CVA Case 1 (Junior): ${cva_case1:.2f}")
print(f"CVA Case 2 (Pari Passu): ${cva_case2:.2f}")



import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# Data for visualization
time_steps = [1, 2, 3]
exposures = [10000, 12000, 8000]
default_probs = [0.02, 0.03, 0.04]
recovery_rate = 0.4
break_date = 2
threshold = 3000
thresholds_by_rating = {'A': 5000, 'BBB': 2000, 'BB': 0}
transition_probs = {'A': 0.7, 'BBB': 0.2, 'BB': 0.1}
derivative_values = [20000, 20000, 20000]
loan_principal = 100000
security_values = [80000, 80000, 80000]

# Calculate CVA for all scenarios
cva_no_break, cva_with_break = calculate_cva_break_clause(exposures, default_probs, recovery_rate, break_date)
cva_no_csa, cva_with_csa = calculate_cva_csa_simple(exposures, default_probs, recovery_rate, threshold)
cva_downgrade = calculate_cva_downgrade_trigger(exposures, default_probs, recovery_rate, thresholds_by_rating, transition_probs)
cva_case1, cva_case2 = calculate_cva_default_waterfall(derivative_values, loan_principal, security_values, default_probs)

# Exposure profiles for visualization
exposures_with_break = exposures.copy()
for i in range(break_date, len(exposures)):
    exposures_with_break[i] = 0

exposures_with_csa = [min(max(exp, 0), threshold) for exp in exposures]

weighted_exposures = [0] * len(exposures)
for i, exp in enumerate(exposures):
    for rating in thresholds_by_rating.keys():
        capped_exp = min(max(exp, 0), thresholds_by_rating[rating])
        weighted_exposures[i] += transition_probs[rating] * capped_exp

# Dash layout
app.layout = html.Div([
    html.H1("Credit Mitigants Impact on Exposure and CVA"),
    
    html.H3("Exposure Profiles"),
    dcc.Graph(id='exposure-graph'),
    
    html.H3("CVA Comparison"),
    dcc.Graph(id='cva-graph'),
])

# Callback to update graphs
@app.callback(
    [Output('exposure-graph', 'figure'),
     Output('cva-graph', 'figure')],
    [Input('exposure-graph', 'id')]
)
def update_graphs(_):
    # Exposure graph
    exposure_fig = go.Figure()
    exposure_fig.add_trace(go.Scatter(x=time_steps, y=exposures, mode='lines+markers', name='No Mitigant'))
    exposure_fig.add_trace(go.Scatter(x=time_steps, y=exposures_with_break, mode='lines+markers', name='With Break Clause'))
    exposure_fig.add_trace(go.Scatter(x=time_steps, y=exposures_with_csa, mode='lines+markers', name='With CSA'))
    exposure_fig.add_trace(go.Scatter(x=time_steps, y=weighted_exposures, mode='lines+markers', name='With Downgrade Trigger'))
    exposure_fig.update_layout(title='Exposure Profiles Over Time', xaxis_title='Time (Years)', yaxis_title='Exposure ($)')

    # CVA graph
    cva_fig = go.Figure()
    cva_values = [cva_no_break, cva_with_break, cva_no_csa, cva_with_csa, cva_downgrade, cva_case1, cva_case2]
    cva_labels = ['No Break', 'With Break', 'No CSA', 'With CSA', 'Downgrade Trigger', 'Waterfall Case 1', 'Waterfall Case 2']
    cva_fig.add_trace(go.Bar(x=cva_labels, y=cva_values))
    cva_fig.update_layout(title='CVA Comparison Across Mitigants', xaxis_title='Scenario', yaxis_title='CVA ($)')

    return exposure_fig, cva_fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)