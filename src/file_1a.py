import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go

# Function to simulate Brownian motion
def simulate_brownian_motion(T=1, N=100, num_paths=5):
    dt = T / N
    t = np.linspace(0, T, N)
    dW = np.random.normal(0, np.sqrt(dt), (num_paths, N))
    W = np.cumsum(dW, axis=1)
    return t, W

# Function for nested Monte Carlo
def nested_monte_carlo(T=1, s=0.5, N_outer=1000, N_inner=100):
    _, W_outer = simulate_brownian_motion(T=s, N=int(s*100), num_paths=N_outer)
    W_s = W_outer[:, -1]
    E_Wt_given_Fs = []
    for w in W_s:
        _, W_inner = simulate_brownian_motion(T=T-s, N=int((T-s)*100), num_paths=N_inner)
        W_t = w + W_inner[:, -1]
        E_Wt_given_Fs.append(np.mean(W_t))
    return W_s, np.array(E_Wt_given_Fs)

# Function for Girsanov transform
def girsanov_transform(T=1, N=100, mu_P=0.2, mu_Q=0.05):
    dt = T / N
    t = np.linspace(0, T, N)
    dW = np.random.normal(0, np.sqrt(dt), N)
    S_P = np.cumsum(mu_P * dt + dW)
    S_Q = np.cumsum(mu_Q * dt + dW)
    return t, S_P, S_Q

# Function for option pricing
def nested_option_pricing(S0=100, K=110, r=0.05, T=1, s=0.5, N_outer=1000, N_inner=100):
    t, S = simulate_brownian_motion(T=s, N=int(s*100), num_paths=N_outer)
    S_s = S[:, -1]
    payoffs = []
    for s_val in S_s:
        _, W_inner = simulate_brownian_motion(T=T-s, N=int((T-s)*100), num_paths=N_inner)
        S_T = s_val * np.exp((r - 0.5*0.2**2)*(T-s) + 0.2*np.cumsum(W_inner, axis=1)[:, -1])
        payoff = np.maximum(S_T - K, 0)
        payoffs.append(np.mean(payoff) * np.exp(-r*(T-s)))
    option_price = np.mean(payoffs)
    return t, S, option_price

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Financial Engineering Dashboard", style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.H3("Brownian Motion Simulation"),
            html.Label("Number of Paths:"),
            dcc.Input(id='num_paths', type='number', value=5),
            html.Label("Time Steps (N):"),
            dcc.Input(id='N_brownian', type='number', value=100),
            html.Button('Run Brownian Motion', id='run_brownian', n_clicks=0),
        ], style={'width': '20%', 'display': 'inline-block'}),

        html.Div([
            html.H3("Nested Monte Carlo Simulation"),
            html.Label("Outer Paths (N_outer):"),
            dcc.Input(id='N_outer', type='number', value=1000),
            html.Label("Inner Paths (N_inner):"),
            dcc.Input(id='N_inner', type='number', value=100),
            html.Button('Run Nested Monte Carlo', id='run_nested', n_clicks=0),
        ], style={'width': '20%', 'display': 'inline-block'}),

        html.Div([
            html.H3("Girsanov Transform"),
            html.Label("Drift (P):"),
            dcc.Input(id='mu_P', type='number', value=0.2),
            html.Label("Drift (Q):"),
            dcc.Input(id='mu_Q', type='number', value=0.05),
            html.Button('Run Girsanov Transform', id='run_girsanov', n_clicks=0),
        ], style={'width': '20%', 'display': 'inline-block'}),

        html.Div([
            html.H3("Option Pricing"),
            html.Label("Initial Stock Price (S0):"),
            dcc.Input(id='S0', type='number', value=100),
            html.Label("Strike Price (K):"),
            dcc.Input(id='K', type='number', value=110),
            html.Label("Risk-Free Rate (r):"),
            dcc.Input(id='r', type='number', value=0.05),
            html.Button('Run Option Pricing', id='run_option', n_clicks=0),
        ], style={'width': '20%', 'display': 'inline-block'}),
    ]),

    html.Div([
        html.Div([
            html.H4("Brownian Motion Simulation"),
            html.P("Brownian motion is a stochastic process with independent, normally distributed increments. It is used to model random movements in financial markets."),
            dcc.Graph(id='brownian_graph'),
            html.P("Interpretation: The graph shows multiple paths of Brownian motion. The spread of paths increases over time, reflecting increasing uncertainty (volatility scales with √t)."),
        ]),
        html.Div([
            html.H4("Nested Monte Carlo Simulation"),
            html.P("Nested Monte Carlo is used to calculate conditional expectations by simulating sub-paths. It is useful for pricing derivatives with path-dependent features."),
            dcc.Graph(id='nested_graph'),
            html.P("Interpretation: The scatter plot shows the conditional expectation E[W_T | F_s] compared to the theoretical value W_s. The points should align with the red line, confirming the martingale property."),
        ]),
        html.Div([
            html.H4("Girsanov Transform"),
            html.P("The Girsanov theorem allows us to change the probability measure from the real-world measure (P) to the risk-neutral measure (Q). This is essential for derivative pricing."),
            dcc.Graph(id='girsanov_graph'),
            html.P("Interpretation: The graph compares the process under the real-world measure (P) and the risk-neutral measure (Q). The risk-neutral process has a reduced drift, reflecting the absence of risk premiums."),
        ]),
        html.Div([
            html.H4("Option Pricing"),
            html.P("Nested Monte Carlo is used to price European call options by simulating future stock prices and calculating the expected payoff under the risk-neutral measure."),
            dcc.Graph(id='option_graph'),
            html.P(f"Interpretation: The graph shows stock price paths up to s=0.5. The option price is calculated using nested Monte Carlo, reflecting the expected payoff max(S_T - K, 0) discounted to the present value."),
        ]),
    ]),
])

# Callbacks for interactivity
@app.callback(
    Output('brownian_graph', 'figure'),
    Input('run_brownian', 'n_clicks'),
    Input('num_paths', 'value'),
    Input('N_brownian', 'value'),
)
def update_brownian(n_clicks, num_paths, N):
    t, W = simulate_brownian_motion(T=1, N=N, num_paths=num_paths)
    fig = go.Figure()
    for i in range(W.shape[0]):
        fig.add_trace(go.Scatter(x=t, y=W[i], mode='lines', name=f'Path {i+1}'))
    fig.update_layout(title="Brownian Motion Paths", xaxis_title="Time (t)", yaxis_title="W(t)")
    return fig

@app.callback(
    Output('nested_graph', 'figure'),
    Input('run_nested', 'n_clicks'),
    Input('N_outer', 'value'),
    Input('N_inner', 'value'),
)
def update_nested(n_clicks, N_outer, N_inner):
    W_s, E_Wt_given_Fs = nested_monte_carlo(T=1, s=0.5, N_outer=N_outer, N_inner=N_inner)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=W_s, y=E_Wt_given_Fs, mode='markers', name='E[W_T | F_s]'))
    fig.add_trace(go.Scatter(x=W_s, y=W_s, mode='lines', name='W_s (Theory)', line=dict(color='red')))
    fig.update_layout(title="Conditional Expectation of Brownian Motion", xaxis_title="W_s", yaxis_title="E[W_T | F_s]")
    return fig

@app.callback(
    Output('girsanov_graph', 'figure'),
    Input('run_girsanov', 'n_clicks'),
    Input('mu_P', 'value'),
    Input('mu_Q', 'value'),
)
def update_girsanov(n_clicks, mu_P, mu_Q):
    t, S_P, S_Q = girsanov_transform(T=1, N=100, mu_P=mu_P, mu_Q=mu_Q)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=S_P, mode='lines', name='Real-World Measure (P)'))
    fig.add_trace(go.Scatter(x=t, y=S_Q, mode='lines', name='Risk-Neutral Measure (Q)'))
    fig.update_layout(title="Measure Change via Girsanov Theorem", xaxis_title="Time (t)", yaxis_title="Process Value")
    return fig

@app.callback(
    Output('option_graph', 'figure'),
    Input('run_option', 'n_clicks'),
    Input('S0', 'value'),
    Input('K', 'value'),
    Input('r', 'value'),
)
def update_option(n_clicks, S0, K, r):
    t, S, option_price = nested_option_pricing(S0=S0, K=K, r=r, T=1, s=0.5, N_outer=1000, N_inner=100)
    fig = go.Figure()
    for i in range(S.shape[0]):
        fig.add_trace(go.Scatter(x=t, y=S[i], mode='lines', name=f'Path {i+1}', line=dict(color='blue', width=0.5)))
    fig.update_layout(title=f"Stock Price Paths up to s=0.5 (Option Price = {option_price:.2f})", xaxis_title="Time (t)", yaxis_title="Stock Price")
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)