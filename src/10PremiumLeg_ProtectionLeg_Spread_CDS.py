import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

# --- Calculation Functions ---

def calculate_premium_leg(notional, spread, times, r, lambda_):
    """
    Calculate the Premium Leg (Formula 4.5) without accrued premium.
    
    Parameters:
    - notional (float): The notional amount of the CDS (e.g., 1,000,000 USD).
    - spread (float): Annual CDS spread in decimal (e.g., 0.02 for 200 bp).
    - times (list): List of payment times in years (e.g., [0.25, 0.5, 0.75, 1.0]).
    - r (float): Constant interest rate (e.g., 0.01 for 1%).
    - lambda_ (float): Constant hazard rate (e.g., 0.02 for 2% annual default risk).
    
    Returns:
    - total_premium (float): Total value of the premium leg.
    - premium_values (list): Individual premium values at each time.
    """
    alpha = 0.25  # Quarterly day-count fraction
    premium_values = []
    for t in times:
        B = np.exp(-r * t)  # Discount factor
        P = np.exp(-lambda_ * t)  # Survival probability
        C = spread * notional * alpha  # Quarterly premium
        V = C * B * P  # Formula 4.4
        premium_values.append(V)
    total_premium = sum(premium_values)  # Formula 4.5
    return total_premium, premium_values

def calculate_protection_leg(notional, recovery, times, r, lambda_):
    """
    Calculate the Protection Leg (Formula 4.14) using a partitioned sum.
    
    Parameters:
    - notional (float): The notional amount of the CDS.
    - recovery (float): Recovery rate (e.g., 0.4 for 40%).
    - times (list): List of partition times (e.g., [0, 0.25, 0.5, 0.75, 1.0]).
    - r (float): Constant interest rate.
    - lambda_ (float): Constant hazard rate.
    
    Returns:
    - total_protection (float): Total value of the protection leg.
    - protection_values (list): Cumulative protection values at each step.
    """
    protection_values = [0]  # Start at 0
    total = 0
    for i in range(1, len(times)):
        u_km1, u_k = times[i-1], times[i]
        B_km1 = np.exp(-r * u_km1)
        B_k = np.exp(-r * u_k)
        P_km1 = np.exp(-lambda_ * u_km1)
        P_k = np.exp(-lambda_ * u_k)
        term = (lambda_ / (lambda_ + r)) * (B_km1 * P_km1 - B_k * P_k)
        total += term * (1 - recovery) * notional
        protection_values.append(total)
    return total, protection_values

def calculate_cds_value(premium, protection, accrued=0):
    """
    Calculate the CDS Value from the buyer's perspective.
    
    Parameters:
    - premium (float): Total premium leg value.
    - protection (float): Total protection leg value.
    - accrued (float, optional): Accrued premium value (default 0 for simplicity).
    
    Returns:
    - cds_value (float): Net value of the CDS.
    """
    return premium + accrued - protection

def calculate_breakeven_spread(notional, recovery, times, r, lambda_):
    """
    Calculate the Breakeven Spread (Formula 4.15, simplified denominator).
    
    Parameters:
    - notional (float): The notional amount of the CDS.
    - recovery (float): Recovery rate.
    - times (list): List of times for premium and protection legs.
    - r (float): Constant interest rate.
    - lambda_ (float): Constant hazard rate.
    
    Returns:
    - S (float): Breakeven spread in decimal.
    """
    _, protection_values = calculate_protection_leg(notional, recovery, times, r, lambda_)
    numerator = protection_values[-1] / notional  # Protection leg / notional
    denominator = sum(np.exp(-r * t) * np.exp(-lambda_ * t) * 0.25 for t in times[1:])
    S = (1 - recovery) * (numerator / denominator)
    return S

# --- Dash App ---
app = Dash(__name__)

app.layout = html.Div([
    html.H1("CDS Valuation Dashboard"),
    html.Div([
        html.Label("Notional Amount (USD):"),
        dcc.Input(id="notional", type="number", value=1000000, step=100000),
        html.Label("Spread (bp):"),
        dcc.Input(id="spread", type="number", value=200, step=10),
        html.Label("Hazard Rate (%):"),
        dcc.Input(id="lambda", type="number", value=2, step=0.1),
        html.Label("Interest Rate (%):"),
        dcc.Input(id="r", type="number", value=1, step=0.1),
        html.Label("Recovery Rate (%):"),
        dcc.Input(id="recovery", type="number", value=40, step=5),
    ]),
    dcc.Graph(id="cds-graph"),
    html.Div(id="results")
])

@app.callback(
    [Output("cds-graph", "figure"), Output("results", "children")],
    [Input("notional", "value"), Input("spread", "value"), Input("lambda", "value"), 
     Input("r", "value"), Input("recovery", "value")]
)
def update_graph(notional, spread, lambda_val, r_val, recovery_val):
    # Default values if inputs are None
    notional = notional if notional is not None else 1000000
    spread = spread if spread is not None else 200
    lambda_val = lambda_val if lambda_val is not None else 2
    r_val = r_val if r_val is not None else 1
    recovery_val = recovery_val if recovery_val is not None else 40

    # Convert inputs to proper units
    spread = spread / 10000  # bp to decimal
    lambda_ = lambda_val / 100
    r = r_val / 100
    recovery = recovery_val / 100
    
    # Fixed times for quarterly payments
    times = [0, 0.25, 0.5, 0.75, 1.0]
    
    # Calculations
    premium, premium_vals = calculate_premium_leg(notional, spread, times[1:], r, lambda_)
    protection, protection_vals = calculate_protection_leg(notional, recovery, times, r, lambda_)
    cds_value = calculate_cds_value(premium, protection)
    breakeven = calculate_breakeven_spread(notional, recovery, times, r, lambda_)
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times[1:], y=premium_vals, mode='lines+markers', name='Premium Leg'))
    fig.add_trace(go.Scatter(x=times, y=protection_vals, mode='lines+markers', name='Protection Leg'))
    fig.update_layout(
        title="CDS Legs Over Time",
        xaxis_title="Time (Years)",
        yaxis_title="Value (USD)",
        template="plotly_white"
    )
    
    # Results with Corrected Interpretations
    results = [
        html.H3("Calculation Results and Interpretations"),
        html.P(f"Total Premium Leg: ${premium:,.2f}"),
        html.P("Interpretation: This is the present value of all premium payments you make to the CDS seller over 1 year, discounted and adjusted for survival probability. It represents your cost for default protection."),
        html.P(f"Total Protection Leg: ${protection:,.2f}"),
        html.P("Interpretation: This is the expected present value of the payout you’d receive if the reference entity defaults, weighted by default probability and discounted. It’s the benefit of the CDS."),
        html.P(f"CDS Value: ${cds_value:,.2f}"),
        html.P(f"Interpretation: From the buyer’s perspective, this is the net value of the CDS (premiums paid minus protection received). A negative value means the premiums are less than the expected protection, suggesting the CDS is a bargain for the buyer at the current spread. You’re paying less than the protection is worth."),
        html.P(f"Breakeven Spread: {breakeven*10000:,.0f} bp"),
        html.P("Interpretation: This is the fair annual spread where the CDS value is zero—premiums equal protection. If the market spread is higher than this, you’re overpaying; if lower (as here), it’s a bargain for the buyer.")
    ]
    
    return fig, results

if __name__ == "__main__":
    app.run(debug=True)