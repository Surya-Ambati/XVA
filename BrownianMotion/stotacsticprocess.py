import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Itô Integral Visualization"),
    html.Div([
        html.Label("Select Time (t):"),
        dcc.Slider(id='time-slider', min=0.1, max=5, step=0.1, value=1, marks={i: str(i) for i in range(0, 6)}),
        html.Button('Simulate Brownian Motion', id='simulate-button', n_clicks=0),
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    html.Div([
        dcc.Graph(id='brownian-motion-plot'),
        dcc.Graph(id='ito-integral-plot'),
    ], style={'width': '65%', 'display': 'inline-block'})
])

# Callback to update the plots
@app.callback(
    [Output('brownian-motion-plot', 'figure'),
     Output('ito-integral-plot', 'figure')],
    [Input('time-slider', 'value'),
     Input('simulate-button', 'n_clicks')]
)
def update_plots(t, n_clicks):
    # Simulate Brownian Motion
    np.random.seed(n_clicks)  # Ensure reproducibility
    time_values = np.linspace(0, t, 1000)
    brownian_motion = np.cumsum(np.random.normal(0, np.sqrt(t / 1000), 1000))

    # Compute the Itô Integral
    ito_integral = 0.5 * brownian_motion**2 - 0.5 * time_values

    # Create Brownian Motion plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=time_values, y=brownian_motion, mode='lines', name='Brownian Motion'))
    fig1.update_layout(
        title=f"Brownian Motion from t=0 to t={t}",
        xaxis_title="Time (t)",
        yaxis_title="Position (B(t))",
        showlegend=True
    )

    # Create Itô Integral plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time_values, y=ito_integral, mode='lines', name='Itô Integral'))
    fig2.update_layout(
        title=f"Itô Integral of Brownian Motion from t=0 to t={t}",
        xaxis_title="Time (t)",
        yaxis_title="Itô Integral",
        showlegend=True
    )

    return fig1, fig2

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)