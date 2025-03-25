import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from scipy.stats import norm
from datetime import datetime, timedelta

# ==============================================
# CVA Calculation Functions
# ==============================================

def simulate_gbm(S0, mu, sigma, T, steps, n_paths):
    """Simulate Geometric Brownian Motion paths.
    
    Parameters:
    S0 (float): Initial stock price
    mu (float): Drift rate (risk-free rate for risk-neutral pricing)
    sigma (float): Volatility
    T (float): Time to maturity (years)
    steps (int): Number of time steps
    n_paths (int): Number of simulation paths
    
    Returns:
    tuple: (time points, simulated paths)
    """
    dt = T / steps
    t = np.linspace(0, T, steps + 1)
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, steps))
    S = np.zeros((n_paths, steps + 1))
    S[:, 0] = S0
    for i in range(1, steps + 1):
        S[:, i] = S[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[:, i-1])
    return t, S

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes option pricing.
    
    Parameters:
    S (float/array): Current stock price(s)
    K (float): Strike price
    T (float): Time to maturity (years)
    r (float): Risk-free rate
    sigma (float): Volatility
    option_type (str): 'call' or 'put'
    
    Returns:
    float/array: Option price(s)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def calculate_cva(S0, K, T, r, sigma, lambda_c, R, steps, n_paths, option_type='call'):
    """Calculate CVA for a European option using Monte Carlo simulation.
    
    Parameters:
    S0 (float): Initial stock price
    K (float): Strike price
    T (float): Time to maturity (years)
    r (float): Risk-free rate
    sigma (float): Volatility
    lambda_c (float): Hazard rate (default intensity)
    R (float): Recovery rate (0-1)
    steps (int): Number of time steps
    n_paths (int): Number of simulation paths
    option_type (str): 'call' or 'put'
    
    Returns:
    tuple: (time points, option values, total CVA, CVA contributions)
    """
    # Simulate stock paths
    t, S = simulate_gbm(S0, r, sigma, T, steps, n_paths)
    
    # Calculate risk-free option values at each time step
    V = np.zeros_like(S)
    for i in range(steps + 1):
        tau = T - t[i]
        V[:, i] = black_scholes(S[:, i], K, tau, r, sigma, option_type)
    
    # Calculate CVA components
    dt = T / steps
    cva = 0
    cva_contributions = np.zeros(steps + 1)
    
    for i in range(steps + 1):
        # Probability of default in [t_i, t_{i+1}]
        prob_default = np.exp(-lambda_c * t[i]) - np.exp(-lambda_c * (t[i] + dt))
        # Discount factor
        discount = np.exp(-r * t[i])
        # Expected positive exposure (EPE)
        if option_type == 'call':
            epe = np.mean(np.maximum(V[:, i], 0))  # For calls, exposure is when V > 0
        else:
            epe = np.mean(np.maximum(-V[:, i], 0))  # For puts, exposure is when counterparty owes us
        # CVA contribution for this time interval
        cva_contrib = (1 - R) * prob_default * discount * epe
        cva += cva_contrib
        cva_contributions[i] = cva_contrib
    
    return t, V, cva, cva_contributions

# ==============================================
# Dash App Layout
# ==============================================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("CVA Calculator (Unilateral Replication)"),
            html.Hr(),
            
            html.H5("Stock Parameters"),
            dbc.InputGroup([
                dbc.InputGroupText("S₀"),
                dbc.Input(id="S0", type="number", value=100)
            ]),
            dbc.InputGroup([
                dbc.InputGroupText("σ"),
                dbc.Input(id="sigma", type="number", value=0.2)
            ]),
            
            html.H5("Option Parameters"),
            dbc.InputGroup([
                dbc.InputGroupText("K"),
                dbc.Input(id="K", type="number", value=100)
            ]),
            dbc.InputGroup([
                dbc.InputGroupText("T (years)"),
                dbc.Input(id="T", type="number", value=1)
            ]),
            dbc.InputGroup([
                dbc.InputGroupText("r (risk-free)"),
                dbc.Input(id="r", type="number", value=0.05)
            ]),
            dbc.InputGroup([
                dbc.InputGroupText("Option Type"),
                dbc.Select(
                    id="option_type",
                    options=[
                        {"label": "Call", "value": "call"},
                        {"label": "Put", "value": "put"}
                    ],
                    value="call"
                )
            ]),
            
            html.H5("Credit Risk Parameters"),
            dbc.InputGroup([
                dbc.InputGroupText("λ (hazard rate)"),
                dbc.Input(id="lambda_c", type="number", value=0.03)
            ]),
            dbc.InputGroup([
                dbc.InputGroupText("R (recovery)"),
                dbc.Input(id="R", type="number", value=0.4)
            ]),
            
            html.H5("Simulation Settings"),
            dbc.InputGroup([
                dbc.InputGroupText("Paths"),
                dbc.Input(id="n_paths", type="number", value=1000)
            ]),
            dbc.InputGroup([
                dbc.InputGroupText("Steps"),
                dbc.Input(id="steps", type="number", value=50)
            ]),
            
            html.Hr(),
            dbc.Button("Calculate CVA", id="calculate", color="primary"),
            html.Div(id="cva_output", className="mt-3"),
            html.Div(id="calculation_steps", className="mt-3")
        ], width=4),
        
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="stock_paths")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="cva_contributions")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="option_values")
                ])
            ])
        ], width=8)
    ])
])

# ==============================================
# Callbacks
# ==============================================

@app.callback(
    [Output("stock_paths", "figure"),
     Output("cva_contributions", "figure"),
     Output("option_values", "figure"),
     Output("cva_output", "children"),
     Output("calculation_steps", "children"),
     Output("simulation_details", "children")],  # New output
    [Input("calculate", "n_clicks")],
    [State("S0", "value"),
     State("K", "value"),
     State("T", "value"),
     State("r", "value"),
     State("sigma", "value"),
     State("lambda_c", "value"),
     State("R", "value"),
     State("n_paths", "value"),
     State("steps", "value"),
     State("option_type", "value")]
)
def update_plots(n_clicks, S0, K, T, r, sigma, lambda_c, R, n_paths, steps, option_type):
    if n_clicks is None:
        return [go.Figure()]*3 + [""]*3
    
    # Get current date and calculate maturity date
    current_date = datetime.now()
    maturity_date = current_date + timedelta(days=T*365)
    
    # Calculate CVA
    t, V, cva, cva_contributions = calculate_cva(S0, K, T, r, sigma, lambda_c, R, steps, n_paths, option_type)
    current_value = black_scholes(S0, K, T, r, sigma, option_type)
    economic_value = current_value - cva
    
    # Simulation details text
    sim_details = html.Div([
        html.H5("Simulation Details"),
        dbc.ListGroup([
            dbc.ListGroupItem(f"Valuation Date: {current_date.strftime('%Y-%m-%d')}"),
            dbc.ListGroupItem(f"Maturity Date: {maturity_date.strftime('%Y-%m-%d')}"),
            dbc.ListGroupItem(f"Time to Maturity: {T} years"),
            dbc.ListGroupItem(f"Simulated Paths: {n_paths}"),
            dbc.ListGroupItem(f"Time Steps: {steps}"),
            dbc.ListGroupItem(f"Final Stock Price Range: ${np.min(V[:,-1]):.2f} - ${np.max(V[:,-1]):.2f}")
        ])
    ])

    # Plot stock paths with improved interpretation
    fig_stock = go.Figure()
    for i in range(min(10, n_paths)):
        fig_stock.add_trace(go.Scatter(x=t, y=V[i], mode='lines', name=f"Path {i+1}"))
    fig_stock.update_layout(
        title=f"{option_type.capitalize()} Option Values Over Time",
        xaxis_title="Years from Today",
        yaxis_title="Option Value ($)",
        margin=dict(b=100)  # Add space for annotation
    )
    fig_stock.add_annotation(
        x=0.5, y=-0.3,
        xref="paper", yref="paper",
        text="Each line shows how the option's value might change over time. "
             "The spread between paths shows the effect of stock price volatility. "
             f"For {option_type} options, values increase when the stock moves {'up' if option_type=='call' else 'down'}.",
        showarrow=False,
        font=dict(size=12),
        align="left"
    )

    # Plot CVA contributions
    fig_cva = go.Figure()
    fig_cva.add_trace(go.Bar(x=t, y=cva_contributions, name="CVA Contribution"))
    fig_cva.update_layout(
        title="Credit Risk Exposure Over Time",
        xaxis_title="Years from Today",
        yaxis_title="CVA Contribution ($)",
        margin=dict(b=100)
    )
    fig_cva.add_annotation(
        x=0.5, y=-0.3,
        xref="paper", yref="paper",
        text="This shows how much credit risk exists at different time periods. "
             "Early periods have more default risk but less money at stake. "
             "Later periods have more money at stake but less time for defaults to occur.",
        showarrow=False,
        font=dict(size=12),
        align="left"
    )

    # Plot option payoffs with simple explanation
    fig_option = go.Figure()
    if option_type == 'call':
        payoff = np.maximum(V[:,-1] - K, 0)
        payoff_title = f"Call Option Payoffs at Maturity"
        payoff_explanation = ("Imagine you have the right to buy at ${K}. "
                            "This shows all possible outcomes - most often the right expires worthless, "
                            "but sometimes the stock goes up and you make money.")
    else:
        payoff = np.maximum(K - V[:,-1], 0)
        payoff_title = f"Put Option Payoffs at Maturity"
        payoff_explanation = ("Imagine insurance against the stock falling below ${K}. "
                            "Most often you don't need it, but when the stock crashes, "
                            "this insurance pays out big.")
    
    fig_option.add_trace(go.Histogram(x=payoff, nbinsx=50, name="Payoffs"))
    fig_option.update_layout(
        title=payoff_title,
        xaxis_title="Potential Profit at Maturity ($)",
        yaxis_title="How Often This Happens",
        margin=dict(b=100)
    )
    fig_option.add_annotation(
        x=0.5, y=-0.3,
        xref="paper", yref="paper",
        text=payoff_explanation,
        showarrow=False,
        font=dict(size=12),
        align="left"
    )

    # Valuation results with clearer presentation
    cva_text = html.Div([
        html.H4("Valuation Results", className="mt-3"),
        dbc.Card([
            dbc.CardHeader("Option Values"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Risk-Free Value", className="card-title"),
                        html.P(f"${current_value:.2f}", className="card-text"),
                        html.Small("What the option would be worth with no default risk", className="text-muted")
                    ]),
                    dbc.Col([
                        html.H6("CVA Adjustment", className="card-title"),
                        html.P(f"-${cva:.2f}", className="card-text text-danger"),
                        html.Small("Expected loss from counterparty default", className="text-muted")
                    ]),
                    dbc.Col([
                        html.H6("Economic Value", className="card-title"),
                        html.P(f"${economic_value:.2f}", className="card-text text-success"),
                        html.Small("What the option is really worth", className="text-muted")
                    ])
                ])
            ])
        ]),
        
        html.H5("What Affects CVA?", className="mt-3"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Parameter"), 
                html.Th("Your Input"), 
                html.Th("Effect on CVA")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td("Hazard Rate (λ)"), 
                    html.Td(lambda_c), 
                    html.Td("Higher → More CVA (more likely to default)")
                ]),
                html.Tr([
                    html.Td("Recovery (R)"), 
                    html.Td(f"{R*100}%"), 
                    html.Td("Higher → Less CVA (get more back if default)")
                ]),
                html.Tr([
                    html.Td("Volatility (σ)"), 
                    html.Td(f"{sigma*100}%"), 
                    html.Td("Higher → More CVA (bigger potential losses)")
                ])
            ])
        ], bordered=True, hover=True, className="mt-2")
    ])

    # Calculation steps with simple language
    steps_text = html.Div([
        html.H4("How We Calculated This", className="mt-3"),
        html.Div([
            html.Div([
                html.Span("1", className="badge bg-primary me-2"),
                "Simulated thousands of possible stock price paths"
            ], className="mb-2"),
            html.Div([
                html.Span("2", className="badge bg-primary me-2"),
                f"Calculated {option_type} option value at each future date"
            ], className="mb-2"),
            html.Div([
                html.Span("3", className="badge bg-primary me-2"),
                "Estimated when defaults might happen and how much we'd lose"
            ], className="mb-2"),
            html.Div([
                html.Span("4", className="badge bg-primary me-2"),
                "Added up all potential losses to get the CVA"
            ], className="mb-2"),
            html.Div([
                html.Span("5", className="badge bg-primary me-2"),
                "Subtracted CVA from the risk-free value"
            ], className="mb-2")
        ], className="ps-3"),
        html.P("Note: This assumes we can't recover anything beyond the recovery rate if the counterparty defaults.", 
              className="text-muted mt-2")
    ])

    return fig_stock, fig_cva, fig_option, cva_text, steps_text, sim_details

# Add the new output to layout
app.layout.children[0].children[1].children.insert(0, 
    dbc.Row([
        dbc.Col(html.Div(id="simulation_details"), width=12)
    ])
)

if __name__ == "__main__":
    app.run(debug=True)@app.callback(
    [Output("stock_paths", "figure"),
     Output("cva_contributions", "figure"),
     Output("option_values", "figure"),
     Output("cva_output", "children"),
     Output("calculation_steps", "children"),
     Output("simulation_details", "children")],  # New output
    [Input("calculate", "n_clicks")],
    [State("S0", "value"),
     State("K", "value"),
     State("T", "value"),
     State("r", "value"),
     State("sigma", "value"),
     State("lambda_c", "value"),
     State("R", "value"),
     State("n_paths", "value"),
     State("steps", "value"),
     State("option_type", "value")]
)
def update_plots(n_clicks, S0, K, T, r, sigma, lambda_c, R, n_paths, steps, option_type):
    if n_clicks is None:
        return [go.Figure()]*3 + [""]*3
    
    # Get current date and calculate maturity date
    current_date = datetime.now()
    maturity_date = current_date + timedelta(days=T*365)
    
    # Calculate CVA
    t, V, cva, cva_contributions = calculate_cva(S0, K, T, r, sigma, lambda_c, R, steps, n_paths, option_type)
    current_value = black_scholes(S0, K, T, r, sigma, option_type)
    economic_value = current_value - cva
    
    # Simulation details text
    sim_details = html.Div([
        html.H5("Simulation Details"),
        dbc.ListGroup([
            dbc.ListGroupItem(f"Valuation Date: {current_date.strftime('%Y-%m-%d')}"),
            dbc.ListGroupItem(f"Maturity Date: {maturity_date.strftime('%Y-%m-%d')}"),
            dbc.ListGroupItem(f"Time to Maturity: {T} years"),
            dbc.ListGroupItem(f"Simulated Paths: {n_paths}"),
            dbc.ListGroupItem(f"Time Steps: {steps}"),
            dbc.ListGroupItem(f"Final Stock Price Range: ${np.min(V[:,-1]):.2f} - ${np.max(V[:,-1]):.2f}")
        ])
    ])

    # Plot stock paths with improved interpretation
    fig_stock = go.Figure()
    for i in range(min(10, n_paths)):
        fig_stock.add_trace(go.Scatter(x=t, y=V[i], mode='lines', name=f"Path {i+1}"))
    fig_stock.update_layout(
        title=f"{option_type.capitalize()} Option Values Over Time",
        xaxis_title="Years from Today",
        yaxis_title="Option Value ($)",
        margin=dict(b=100)  # Add space for annotation
    )
    fig_stock.add_annotation(
        x=0.5, y=-0.3,
        xref="paper", yref="paper",
        text="Each line shows how the option's value might change over time. "
             "The spread between paths shows the effect of stock price volatility. "
             f"For {option_type} options, values increase when the stock moves {'up' if option_type=='call' else 'down'}.",
        showarrow=False,
        font=dict(size=12),
        align="left"
    )

    # Plot CVA contributions
    fig_cva = go.Figure()
    fig_cva.add_trace(go.Bar(x=t, y=cva_contributions, name="CVA Contribution"))
    fig_cva.update_layout(
        title="Credit Risk Exposure Over Time",
        xaxis_title="Years from Today",
        yaxis_title="CVA Contribution ($)",
        margin=dict(b=100)
    )
    fig_cva.add_annotation(
        x=0.5, y=-0.3,
        xref="paper", yref="paper",
        text="This shows how much credit risk exists at different time periods. "
             "Early periods have more default risk but less money at stake. "
             "Later periods have more money at stake but less time for defaults to occur.",
        showarrow=False,
        font=dict(size=12),
        align="left"
    )

    # Plot option payoffs with simple explanation
    fig_option = go.Figure()
    if option_type == 'call':
        payoff = np.maximum(V[:,-1] - K, 0)
        payoff_title = f"Call Option Payoffs at Maturity"
        payoff_explanation = ("Imagine you have the right to buy at ${K}. "
                            "This shows all possible outcomes - most often the right expires worthless, "
                            "but sometimes the stock goes up and you make money.")
    else:
        payoff = np.maximum(K - V[:,-1], 0)
        payoff_title = f"Put Option Payoffs at Maturity"
        payoff_explanation = ("Imagine insurance against the stock falling below ${K}. "
                            "Most often you don't need it, but when the stock crashes, "
                            "this insurance pays out big.")
    
    fig_option.add_trace(go.Histogram(x=payoff, nbinsx=50, name="Payoffs"))
    fig_option.update_layout(
        title=payoff_title,
        xaxis_title="Potential Profit at Maturity ($)",
        yaxis_title="How Often This Happens",
        margin=dict(b=100)
    )
    fig_option.add_annotation(
        x=0.5, y=-0.3,
        xref="paper", yref="paper",
        text=payoff_explanation,
        showarrow=False,
        font=dict(size=12),
        align="left"
    )

    # Valuation results with clearer presentation
    cva_text = html.Div([
        html.H4("Valuation Results", className="mt-3"),
        dbc.Card([
            dbc.CardHeader("Option Values"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Risk-Free Value", className="card-title"),
                        html.P(f"${current_value:.2f}", className="card-text"),
                        html.Small("What the option would be worth with no default risk", className="text-muted")
                    ]),
                    dbc.Col([
                        html.H6("CVA Adjustment", className="card-title"),
                        html.P(f"-${cva:.2f}", className="card-text text-danger"),
                        html.Small("Expected loss from counterparty default", className="text-muted")
                    ]),
                    dbc.Col([
                        html.H6("Economic Value", className="card-title"),
                        html.P(f"${economic_value:.2f}", className="card-text text-success"),
                        html.Small("What the option is really worth", className="text-muted")
                    ])
                ])
            ])
        ]),
        
        html.H5("What Affects CVA?", className="mt-3"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Parameter"), 
                html.Th("Your Input"), 
                html.Th("Effect on CVA")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td("Hazard Rate (λ)"), 
                    html.Td(lambda_c), 
                    html.Td("Higher → More CVA (more likely to default)")
                ]),
                html.Tr([
                    html.Td("Recovery (R)"), 
                    html.Td(f"{R*100}%"), 
                    html.Td("Higher → Less CVA (get more back if default)")
                ]),
                html.Tr([
                    html.Td("Volatility (σ)"), 
                    html.Td(f"{sigma*100}%"), 
                    html.Td("Higher → More CVA (bigger potential losses)")
                ])
            ])
        ], bordered=True, hover=True, className="mt-2")
    ])

    # Calculation steps with simple language
    steps_text = html.Div([
        html.H4("How We Calculated This", className="mt-3"),
        html.Div([
            html.Div([
                html.Span("1", className="badge bg-primary me-2"),
                "Simulated thousands of possible stock price paths"
            ], className="mb-2"),
            html.Div([
                html.Span("2", className="badge bg-primary me-2"),
                f"Calculated {option_type} option value at each future date"
            ], className="mb-2"),
            html.Div([
                html.Span("3", className="badge bg-primary me-2"),
                "Estimated when defaults might happen and how much we'd lose"
            ], className="mb-2"),
            html.Div([
                html.Span("4", className="badge bg-primary me-2"),
                "Added up all potential losses to get the CVA"
            ], className="mb-2"),
            html.Div([
                html.Span("5", className="badge bg-primary me-2"),
                "Subtracted CVA from the risk-free value"
            ], className="mb-2")
        ], className="ps-3"),
        html.P("Note: This assumes we can't recover anything beyond the recovery rate if the counterparty defaults.", 
              className="text-muted mt-2")
    ])

    return fig_stock, fig_cva, fig_option, cva_text, steps_text, sim_details

# Add the new output to layout
app.layout.children[0].children[1].children.insert(0, 
    dbc.Row([
        dbc.Col(html.Div(id="simulation_details"), width=12)
    ])
)

if __name__ == "__main__":
    app.run(debug=True)