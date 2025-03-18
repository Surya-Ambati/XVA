import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
from scipy.stats import norm

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Brownian Motion and Martingales Visualization"),
    html.Div([
        html.Label("Select Time (t):"),
        dcc.Slider(id='time-slider', min=0.1, max=5, step=0.1, value=1, marks={i: str(i) for i in range(0, 6)}),
        html.Label("Select Level (m):"),
        dcc.Slider(id='level-slider', min=0.1, max=5, step=0.1, value=1, marks={i: str(i) for i in range(0, 6)}),
        html.Button('Simulate Brownian Motion', id='simulate-button', n_clicks=0),
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    html.Div([
        dcc.Graph(id='brownian-motion-plot'),
        html.Div(id='martingale-definition'),
        html.Div(id='martingale-interpretation'),
        dcc.Graph(id='stopping-time-plot'),
        html.Div(id='stopping-time-definition'),
        html.Div(id='stopping-time-interpretation'),
    ], style={'width': '65%', 'display': 'inline-block'})
])

# Callback to update the plots and definitions
@app.callback(
    [Output('brownian-motion-plot', 'figure'),
     Output('martingale-definition', 'children'),
     Output('martingale-interpretation', 'children'),
     Output('stopping-time-plot', 'figure'),
     Output('stopping-time-definition', 'children'),
     Output('stopping-time-interpretation', 'children')],
    [Input('time-slider', 'value'),
     Input('level-slider', 'value'),
     Input('simulate-button', 'n_clicks')]
)
def update_plots(t, m, n_clicks):
    # Simulate Brownian Motion
    np.random.seed(n_clicks)  # Ensure reproducibility
    time_values = np.linspace(0, t, 1000)
    brownian_motion = np.cumsum(np.random.normal(0, np.sqrt(t / 1000), 1000))

    # Create Brownian Motion plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=time_values, y=brownian_motion, mode='lines', name='Brownian Motion'))
    fig1.update_layout(
        title=f"Brownian Motion from t=0 to t={t}",
        xaxis_title="Time (t)",
        yaxis_title="Position (B(t))",
        showlegend=True
    )

    # Martingale Definition and Interpretation
    martingale_definition = html.Div([
        html.H3("What is a Martingale?"),
        html.P("A martingale is a stochastic process where the expected future value, given all past information, is equal to the current value. Formally, for a process \( X(t) \):"),
        html.P(r"\[ \mathbb{E}[X(t) \mid \mathcal{F}_s] = X(s) \quad \text{for} \quad s < t \]"),
        html.P("Brownian Motion \( B(t) \) is a classic example of a martingale.")
    ])
    martingale_interpretation = html.Div([
        html.H3("Interpretation of Brownian Motion as a Martingale"),
        html.P("The graph shows a simulated path of Brownian Motion. The process is a martingale because its future movements are unpredictable given its past. The expected value of \( B(t) \) at any future time is equal to its current value.")
    ])

    # Stopping Time Plot
    hitting_time = np.argmax(brownian_motion >= m) / 1000 * t if np.any(brownian_motion >= m) else t
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time_values, y=brownian_motion, mode='lines', name='Brownian Motion'))
    fig2.add_trace(go.Scatter(x=[hitting_time], y=[m], mode='markers', marker=dict(size=10, color='red'), name=f'First Hitting Time: {hitting_time:.2f}'))
    fig2.update_layout(
        title=f"First Hitting Time of Level m={m}",
        xaxis_title="Time (t)",
        yaxis_title="Position (B(t))",
        showlegend=True
    )

    # Stopping Time Definition and Interpretation
    stopping_time_definition = html.Div([
        html.H3("What is a Stopping Time?"),
        html.P("A stopping time \( \tau \) is a random time that depends only on the information available up to that time. For example, the first time Brownian Motion hits a level \( m \):"),
        html.P(r"\[ \tau_m = \inf \{ t \geq 0 : B(t) = m \} \]")
    ])
    stopping_time_interpretation = html.Div([
        html.H3("Interpretation of Stopping Times"),
        html.P(f"The red dot on the graph shows the first time the Brownian Motion hits the level \( m = {m} \). The stopping time \( \tau_m \) is finite with probability 1, but its expectation is infinite.")
    ])

    return fig1, martingale_definition, martingale_interpretation, fig2, stopping_time_definition, stopping_time_interpretation

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)