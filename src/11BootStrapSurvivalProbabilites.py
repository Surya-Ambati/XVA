import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import newton
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

# --- Hazard Rate Models ---

def piecewise_constant_hazard(times, lambdas, t_max=None):
    t_max = times[-1] if t_max is None else t_max
    t = np.linspace(0, t_max, 1000)  # Increased resolution
    lambda_t = np.zeros_like(t)
    for i in range(len(times) - 1):
        mask = (t >= times[i]) & (t < times[i+1])
        lambda_idx = min(i, len(lambdas) - 1)
        lambda_t[mask] = lambdas[lambda_idx] if lambda_idx >= 0 else 0
    lambda_idx = len(lambdas) - 1 if lambdas else -1
    lambda_t[t >= times[-1]] = lambdas[lambda_idx] if lambda_idx >= 0 else 0
    return t, lambda_t

def piecewise_linear_hazard(times, lambdas, t_max=None):
    t_max = times[-1] if t_max is None else t_max
    t = np.linspace(0, t_max, 1000)
    if len(lambdas) < len(times):
        last_lambda = lambdas[-1] if lambdas else 0
        extended_lambdas = lambdas + [last_lambda] * (len(times) - len(lambdas))
    else:
        extended_lambdas = lambdas[:len(times)]
    lambda_t = np.interp(t, times, extended_lambdas)
    return t, lambda_t

def cubic_spline_hazard(times, lambdas, t_max=None):
    t_max = times[-1] if t_max is None else t_max
    t = np.linspace(0, t_max, 1000)
    if len(lambdas) < len(times):
        last_lambda = lambdas[-1] if lambdas else 0
        extended_lambdas = lambdas + [last_lambda] * (len(times) - len(lambdas))
    else:
        extended_lambdas = lambdas[:len(times)]
    cs = CubicSpline(times, extended_lambdas, bc_type='natural')
    lambda_t = cs(t)
    lambda_t[lambda_t < 0] = 0
    return t, lambda_t

# --- Survival Probability and CDS Valuation ---

def survival_probability(times, lambdas, model='piecewise_constant', t_max=None):
    t_max = times[-1] if t_max is None else t_max
    t = np.linspace(0, t_max, 1000)
    if model == 'piecewise_constant':
        _, lambda_t = piecewise_constant_hazard(times, lambdas, t_max)
    elif model == 'piecewise_linear':
        _, lambda_t = piecewise_linear_hazard(times, lambdas, t_max)
    else:
        _, lambda_t = cubic_spline_hazard(times, lambdas, t_max)
    
    cum_hazard = np.zeros_like(t)
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        cum_hazard[i] = cum_hazard[i-1] + lambda_t[i-1] * dt
    
    P = np.exp(-cum_hazard)
    return t, P

def initial_survival_probability(t_max, lambda_guess=0.01):
    t = np.linspace(0, t_max, 1000)
    P = np.exp(-lambda_guess * t)
    return t, P

def cds_value(notional, spread, times, r, lambdas, model='piecewise_constant'):
    alpha = 0.25
    recovery = 0.4
    premium_leg = 0
    t_fine, P_fine = survival_probability(times, lambdas, model, t_max=times[-1])
    
    # Premium Leg
    for t in times[1:]:
        B = np.exp(-r * t)
        P = np.interp(t, t_fine, P_fine)
        C = spread * notional * alpha
        premium_leg += C * B * P
    
    # Protection Leg (corrected to match book methodology)
    protection_leg = 0
    for i in range(len(times) - 1):
        t_start, t_end = times[i], times[i+1]
        lambda_i = lambdas[min(i, len(lambdas)-1)] if lambdas else 0
        if lambda_i == 0:
            continue
        # Survival probabilities at segment boundaries
        P_start = np.interp(t_start, t_fine, P_fine)
        P_end = np.interp(t_end, t_fine, P_fine)
        B_start = np.exp(-r * t_start)
        B_end = np.exp(-r * t_end)
        term = (lambda_i / (lambda_i + r)) * (B_start * P_start - B_end * P_end)
        protection_leg += (1 - recovery) * notional * term
    
    cds_val = premium_leg - protection_leg
    return premium_leg, protection_leg, cds_val

def bootstrap_survival_probabilities(notional, spreads, maturities, r):
    times = [0] + maturities
    lambdas = []
    steps = []
    
    for i in range(len(maturities)):
        t_start = times[i]
        t_end = times[i+1]
        spread = spreads[i]
        
        # Define payment times within this segment
        payment_times = [t for t in np.arange(t_start + 0.25, t_end + 0.25, 0.25) if t <= t_end]
        if not payment_times:
            payment_times = [t_end]
        segment_times = [0] + payment_times
        
        def cds_value_for_lambda(lambda_i):
            current_lambdas = lambdas + [lambda_i]
            _, _, cds_val = cds_value(notional, spread, segment_times, r, current_lambdas)
            return cds_val
        
        # Use Newton-Raphson to find lambda_i
        try:
            lambda_i = newton(cds_value_for_lambda, 0.01, tol=1e-6, maxiter=100)
            if lambda_i < 0:
                lambda_i = 0.001
        except RuntimeError:
            lambda_i = 0.01
        lambdas.append(lambda_i)
        
        # Calculate survival probability at t_end using all lambdas up to this point
        t_fine, P_fine = survival_probability(times[:i+2], lambdas, 'piecewise_constant')
        P_end = np.interp(t_end, t_fine, P_fine)
        
        # Log the step
        step = f"Segment {i+1} ({t_start}–{t_end} years): Spread = {spread*10000:.0f} bp, " \
               f"Lambda = {lambda_i:.4f}, Survival Probability at {t_end} years = {P_end:.4f}"
        steps.append(step)
    
    return times, lambdas, steps

# --- Dash App ---

app = Dash(__name__)

# CSS for light and dark themes
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

default_spreads = [46, 52, 71, 94, 137, 158, 173, 176, 179, 178]
default_maturities = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]

app.layout = html.Div([
    html.Div([
        html.H1("Hazard Rate and Survival Probability Dashboard", style={'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id="theme-toggle",
                options=[
                    {'label': 'Light', 'value': 'plotly_white'},
                    {'label': 'Dark', 'value': 'plotly_dark'}
                ],
                value='plotly_white',
                clearable=False,
                style={'width': '100px', 'display': 'inline-block'}
            )
        ], style={'position': 'absolute', 'top': '10px', 'right': '10px'})
    ]),
    html.Div([
        html.Label("Notional Amount (USD):"),
        dcc.Input(id="notional", type="number", value=1000000, step=100000),
        html.Label("Interest Rate (%):"),
        dcc.Input(id="r", type="number", value=1, step=0.1),
        html.Label("Initial Hazard Rate Guess (%):"),
        dcc.Input(id="lambda-guess", type="number", value=2, step=0.1),
        html.Label("CDS Spreads (bp, comma-separated):"),
        dcc.Input(id="spreads", type="text", value="46,52,71,94,137,158,173,176,179,178"),
        html.Label("Maturities (years, comma-separated):"),
        dcc.Input(id="maturities", type="text", value="0.5,1,2,3,5,7,10,15,20,30"),
        html.Label("Hazard Rate Model:"),
        dcc.Dropdown(id="model", options=[
            {'label': 'Piecewise Constant', 'value': 'piecewise_constant'},
            {'label': 'Piecewise Linear', 'value': 'piecewise_linear'},
            {'label': 'Cubic Spline', 'value': 'cubic_spline'}
        ], value='piecewise_constant'),
    ]),
    dcc.Graph(id="hazard-graph"),
    dcc.Graph(id="survival-graph"),
    html.Div(id="results")
], id="main-container")

@app.callback(
    [Output("hazard-graph", "figure"), Output("survival-graph", "figure"), Output("results", "children"),
     Output("main-container", "style")],
    [Input("notional", "value"), Input("r", "value"), Input("lambda-guess", "value"),
     Input("spreads", "value"), Input("maturities", "value"), Input("model", "value"),
     Input("theme-toggle", "value")]
)
def update_graphs(notional, r, lambda_guess, spreads, maturities, model, theme):
    # Handle inputs
    notional = notional if notional is not None else 1000000
    r = (r if r is not None else 1) / 100
    lambda_guess = (lambda_guess if lambda_guess is not None else 2) / 100
    try:
        spreads = [float(s) / 10000 for s in spreads.split(",")]
        maturities = [float(m) for m in maturities.split(",")]
    except:
        spreads = default_spreads
        maturities = default_maturities

    # Initial survival probabilities
    t_initial, P_initial = initial_survival_probability(maturities[-1], lambda_guess)
    
    # Bootstrap hazard rates and survival probabilities
    times, lambdas, steps = bootstrap_survival_probabilities(notional, spreads, maturities, r)
    t_bootstrapped, P_bootstrapped = survival_probability(times, lambdas, model)

    # Hazard rate plot
    if model == 'piecewise_constant':
        t, lambda_t = piecewise_constant_hazard(times, lambdas)
    elif model == 'piecewise_linear':
        t, lambda_t = piecewise_linear_hazard(times, lambdas)
    else:
        t, lambda_t = cubic_spline_hazard(times, lambdas)
    
    hazard_fig = go.Figure()
    hazard_fig.add_trace(go.Scatter(x=t, y=lambda_t, mode='lines', name='Hazard Rate'))
    hazard_fig.update_layout(
        title="Hazard Rate Over Time",
        xaxis_title="Time (Years)",
        yaxis_title="Hazard Rate",
        template=theme
    )

    # Survival probability plot
    survival_fig = go.Figure()
    survival_fig.add_trace(go.Scatter(x=t_initial, y=P_initial, mode='lines', name='Initial Survival Probability'))
    survival_fig.add_trace(go.Scatter(x=t_bootstrapped, y=P_bootstrapped, mode='lines', name='Bootstrapped Survival Probability'))
    survival_fig.update_layout(
        title="Survival Probability Over Time",
        xaxis_title="Time (Years)",
        yaxis_title="P(tau > t)",
        template=theme
    )

    # Results and Interpretations
    results = [
        html.H3("Bootstrapping Steps"),
        html.P("The bootstrapping process solves for hazard rates (λ) for each segment using CDS spreads, setting V_CDS = 0. Steps:"),
        html.Ul([html.Li(step) for step in steps]),
        html.H3("Initial Survival Probabilities"),
        html.P(f"At 1 year: {np.interp(1, t_initial, P_initial):.4f}"),
        html.P(f"At 5 years: {np.interp(5, t_initial, P_initial):.4f}"),
        html.P(f"At 10 years: {np.interp(10, t_initial, P_initial):.4f}"),
        html.P(f"Interpretation: These are rough estimates based on a constant hazard rate guess (λ = {lambda_guess * 100:.2f}%). They assume a uniform default risk and do not reflect market data."),
        html.H3("Bootstrapped Survival Probabilities"),
        html.P(f"At 1 year: {np.interp(1, t_bootstrapped, P_bootstrapped):.4f}"),
        html.P(f"At 5 years: {np.interp(5, t_bootstrapped, P_bootstrapped):.4f}"),
        html.P(f"At 10 years: {np.interp(10, t_bootstrapped, P_bootstrapped):.4f}"),
        html.P("Interpretation: These are market-implied survival probabilities derived from CDS spreads. They reflect the actual default risk priced into the market, making them suitable for pricing CDS contracts and calculating CVA.")
    ]

    # Theme styling
    container_style = {
        'backgroundColor': '#ffffff' if theme == 'plotly_white' else '#1a1a1a',
        'color': '#000000' if theme == 'plotly_white' else '#ffffff',
        'padding': '20px'
    }

    return hazard_fig, survival_fig, results, container_style

if __name__ == "__main__":
    app.run(debug=True)