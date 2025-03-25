import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from scipy.stats import norm
from datetime import datetime, timedelta
import textwrap

# ==============================================
# CVA Calculation Functions
# ==============================================

def simulate_gbm(S0, mu, sigma, T, steps, n_paths):
    """Simulate Geometric Brownian Motion paths."""
    dt = T / steps
    t = np.linspace(0, T, steps + 1)
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, steps))
    S = np.zeros((n_paths, steps + 1))
    S[:, 0] = S0
    for i in range(1, steps + 1):
        S[:, i] = S[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[:, i-1])
    return t, S

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes option pricing."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def calculate_cva(S0, K, T, r, sigma, lambda_c, R, steps, n_paths, option_type='call'):
    """Calculate CVA for a European option using Monte Carlo simulation."""
    t, S = simulate_gbm(S0, r, sigma, T, steps, n_paths)
    V = np.zeros_like(S)
    for i in range(steps + 1):
        tau = T - t[i]
        V[:, i] = black_scholes(S[:, i], K, tau, r, sigma, option_type)
    
    dt = T / steps
    cva = 0
    cva_contributions = np.zeros(steps + 1)
    
    for i in range(steps + 1):
        # Probability of default between t[i] and t[i] + dt
        if i < steps:
            prob_default = (1 - np.exp(-lambda_c * dt)) * np.exp(-lambda_c * t[i])
        else:
            prob_default = (1 - np.exp(-lambda_c * (T - t[i]))) * np.exp(-lambda_c * t[i])
        # Ensure prob_default is non-negative
        prob_default = max(prob_default, 0)
        # Discount factor at time t[i]
        discount = np.exp(-r * t[i])
        # Expected Positive Exposure (EPE)
        if option_type == 'call':
            epe = np.mean(np.maximum(V[:, i], 0))
        else:
            epe = np.mean(np.maximum(-V[:, i], 0))
        # CVA contribution at each time step
        cva_contrib = (1 - R) * prob_default * discount * epe
        cva += cva_contrib
        cva_contributions[i] = cva_contrib
    
    return t, S, V, cva, cva_contributions

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
            dbc.InputGroup([
                dbc.InputGroupText("Paths to Display"),
                dbc.Input(id="display_paths", type="number", value=100, min=1, max=1000)
            ]),
            
            html.Hr(),
            dbc.Button("Calculate CVA", id="calculate", color="primary"),
            html.Div(id="cva_output", className="mt-3"),
            html.Div(id="calculation_steps", className="mt-3")
        ], width=4),
        
        dbc.Col([
            dbc.Row([
                dbc.Col(html.Div(id="simulation_details"), width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="stock_paths", style={'width': '100%', 'height': '600px', 'padding': '20px'})
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="cva_contributions", style={'width': '100%', 'height': '600px', 'padding': '20px'})
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="option_values", style={'width': '100%', 'height': '600px', 'padding': '20px'})
                ])
            ])
        ], width=8)
    ])
], fluid=True)

# ==============================================
# Callbacks
# ==============================================

@app.callback(
    [Output("stock_paths", "figure"),
     Output("cva_contributions", "figure"),
     Output("option_values", "figure"),
     Output("cva_output", "children"),
     Output("calculation_steps", "children"),
     Output("simulation_details", "children")],
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
     State("option_type", "value"),
     State("display_paths", "value")]
)
def update_plots(n_clicks, S0, K, T, r, sigma, lambda_c, R, n_paths, steps, option_type, display_paths):
    if n_clicks is None:
        return [go.Figure()]*3 + [""]*3
    
    # Ensure inputs are valid
    display_paths = min(max(1, int(display_paths or 10)), n_paths)
    
    # Calculate CVA
    t, S, V, cva, cva_contributions = calculate_cva(S0, K, T, r, sigma, lambda_c, R, steps, n_paths, option_type)
    current_value = black_scholes(S0, K, T, r, sigma, option_type)
    economic_value = current_value - cva
    
    # Simulation details with stock price range
    current_date = datetime.now()
    maturity_date = current_date + timedelta(days=T*365)
    sim_details = html.Div([
        html.H5("Simulation Details"),
        dbc.ListGroup([
            dbc.ListGroupItem(f"Valuation Date: {current_date.strftime('%Y-%m-%d')}"),
            dbc.ListGroupItem(f"Maturity Date: {maturity_date.strftime('%Y-%m-%d')}"),
            dbc.ListGroupItem(f"Time to Maturity: {T} years"),
            dbc.ListGroupItem(f"Simulated Paths: {n_paths}"),
            dbc.ListGroupItem(f"Displayed Paths: {display_paths}"),
            dbc.ListGroupItem(f"Time Steps: {steps}"),
            dbc.ListGroupItem(f"Final Stock Price Range: ${np.min(S[:,-1]):.2f} - ${np.max(S[:,-1]):.2f}")
        ])
    ])

    # Function to wrap text for annotations
    def wrap_text(text, width=60):
        return "<br>".join(textwrap.wrap(text, width=width))

    # Stock paths plot with layman explanation
    fig_stock = go.Figure()
    for i in range(min(display_paths, n_paths)):
        fig_stock.add_trace(go.Scatter(x=t, y=V[i], mode='lines', name=f"Path {i+1}"))
    fig_stock.update_layout(
        title=f"{option_type.capitalize()} Option Values Over Time",
        xaxis_title="Years from Today",
        yaxis_title="Option Value ($)",
        margin=dict(l=100, r=100, t=50, b=300),
        font=dict(size=14),
        autosize=True
    )
    stock_text = wrap_text(
        f"We're showing {display_paths} possible futures out of {n_paths} we imagined. "
        f"Each line is like a story of how your option's value might change over {T} year(s). "
        f"If you have a {option_type} option, it gets more valuable when the stock price goes "
        f"{'up' if option_type=='call' else 'down'}. The different lines show how uncertain the future is!"
    )
    fig_stock.add_annotation(
        x=0.5, y=-0.8,
        xref="paper", yref="paper",
        text=stock_text,
        showarrow=False,
        font=dict(size=16),
        align="center",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        width=600
    )

    # CVA contributions plot with layman explanation
    fig_cva = go.Figure()
    fig_cva.add_trace(go.Bar(x=t, y=cva_contributions, name="CVA Contribution"))
    fig_cva.update_layout(
        title="Credit Risk Exposure Over Time",
        xaxis_title="Years from Today",
        yaxis_title="CVA Contribution ($)",
        margin=dict(l=100, r=100, t=50, b=300),
        font=dict(size=14),
        autosize=True,
        yaxis=dict(range=[0, max(cva_contributions) * 1.1 if max(cva_contributions) > 0 else 1])
    )
    cva_text = wrap_text(
        f"This shows how much money you might lose if the other party can't pay you, "
        f"totaling ${cva:.2f} over {T} year(s). Each bar is a chunk of time. At the start, "
        f"there's a bigger chance they might not pay, but there's less money at risk. Later on, "
        f"there's more money at risk, but less chance of them not paying. We add all these risks "
        f"up to know how much to be careful about!"
    )
    fig_cva.add_annotation(
        x=0.5, y=-1,
        xref="paper", yref="paper",
        text=cva_text,
        showarrow=False,
        font=dict(size=16),
        align="center",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        width=600
    )

    # Option payoffs plot with corrected payoff calculation and updated explanation
    fig_option = go.Figure()
    if option_type == 'call':
        payoff = np.maximum(S[:,-1] - K, 0)
        payoff_title = "Call Option Payoffs at Maturity"
        payoff_text = wrap_text(
            f"This shows how much money you might make in {T} year(s) with your call option. "
            f"A call option lets you buy the stock at ${K}. If the stock price is higher than ${K}, you make money! "
            f"For example, if the stock price is $110, you make $10 per share. "
            f"We imagined {n_paths} possible futures, and this graph shows how often you might make different amounts. "
            f"Most times, you might not make anything (the big bar at 0), but sometimes you could make a lot!"
        )
    else:
        payoff = np.maximum(K - S[:,-1], 0)
        payoff_title = "Put Option Payoffs at Maturity"
        payoff_text = wrap_text(
            f"This shows how much money you might make in {T} year(s) with your put option. "
            f"A put option lets you sell the stock at ${K}. If the stock price is lower than ${K}, you make money! "
            f"For example, if the stock price is $90, you make $10 per share. "
            f"We imagined {n_paths} possible futures, and this graph shows how often you might make different amounts. "
            f"Most times, you might not make anything (the big bar at 0), but sometimes you could make a lot if the stock price drops!"
        )
    
    fig_option.add_trace(go.Histogram(x=payoff, nbinsx=50, name="Payoffs"))
    fig_option.update_layout(
        title=payoff_title,
        xaxis_title="Potential Profit at Maturity ($)",
        yaxis_title="How Often It Might Happen",
        margin=dict(l=100, r=100, t=50, b=300),
        font=dict(size=14),
        autosize=True,
        xaxis=dict(range=[0, max(payoff) * 1.1])
    )
    fig_option.add_annotation(
        x=0.5, y=-1.2,
        xref="paper", yref="paper",
        text=payoff_text,
        showarrow=False,
        font=dict(size=16),
        align="center",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        width=600
    )

    # Valuation results
    cva_text = html.Div([
        html.H4("Valuation Results", className="mt-3"),
        dbc.Card([
            dbc.CardHeader("Option Values"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Risk-Free Value", className="card-title"),
                        html.P(f"${current_value:.2f}", className="card-text"),
                        html.Small("If there were no risks", className="text-muted")
                    ]),
                    dbc.Col([
                        html.H6("CVA Adjustment", className="card-title"),
                        html.P(f"-${cva:.2f}", className="card-text text-danger"),
                        html.Small("Risk of not getting paid", className="text-muted")
                    ]),
                    dbc.Col([
                        html.H6("Economic Value", className="card-title"),
                        html.P(f"${economic_value:.2f}", className="card-text text-success"),
                        html.Small("What it's really worth", className="text-muted")
                    ])
                ])
            ])
        ]),
        
        html.H5("What Affects the Risk?", className="mt-3"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Factor"), 
                html.Th("Your Input"), 
                html.Th("How It Changes Risk")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td("Chance of Not Paying (λ)"), 
                    html.Td(lambda_c), 
                    html.Td("Higher → More Risk")
                ]),
                html.Tr([
                    html.Td("Money Recovered (R)"), 
                    html.Td(f"{R*100}%"), 
                    html.Td("Higher → Less Risk")
                ]),
                html.Tr([
                    html.Td("Stock Price Swings (σ)"), 
                    html.Td(f"{sigma*100}%"), 
                    html.Td("Higher → More Risk")
                ])
            ])
        ], bordered=True, hover=True, className="mt-2")
    ])

    # Calculation steps
    steps_text = html.Div([
        html.H4("How We Figured This Out", className="mt-3"),
        html.Div([
            html.Div([
                html.Span("1", className="badge bg-primary me-2"),
                "Imagined lots of possible stock price futures"
            ], className="mb-2"),
            html.Div([
                html.Span("2", className="badge bg-primary me-2"),
                f"Figured out your {option_type} option's value in each future"
            ], className="mb-2"),
            html.Div([
                html.Span("3", className="badge bg-primary me-2"),
                "Guessed when the other party might not pay and how much you'd lose"
            ], className="mb-2"),
            html.Div([
                html.Span("4", className="badge bg-primary me-2"),
                "Added up all the possible losses to get the risk (CVA)"
            ], className="mb-2"),
            html.Div([
                html.Span("5", className="badge bg-primary me-2"),
                "Took the risk away from the perfect-world value"
            ], className="mb-2")
        ], className="ps-3"),
        html.P("We assumed you only get back a fixed amount if they can't pay.", className="text-muted mt-2")
    ])

    return fig_stock, fig_cva, fig_option, cva_text, steps_text, sim_details

if __name__ == "__main__":
    app.run(debug=True)