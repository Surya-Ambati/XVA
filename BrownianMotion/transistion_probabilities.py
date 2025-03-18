import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
from scipy.stats import norm

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Transition Probabilities and Particle Movement Visualization"),
    html.Div([
        html.Label("Select Time (t):"),
        dcc.Slider(id='time-slider', min=0.1, max=20, step=0.1, value=1, marks={i: str(i) for i in range(0, 21)}),
        html.Label("Select Particle Position Range (Δ):"),
        dcc.RangeSlider(id='range-slider', min=-10, max=10, step=0.1, value=[-1, 1], marks={i: str(i) for i in range(-10, 11)}),
        html.Label("Select Particle Position (y):"),
        dcc.Slider(id='y-slider', min=-10, max=10, step=0.1, value=0, marks={i: str(i) for i in range(-10, 11)}),
        html.Div(id='probability-output', style={'marginTop': '20px', 'fontSize': '16px', 'border': '1px solid #ddd', 'padding': '10px', 'borderRadius': '5px'}),
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    html.Div([
        html.H3("Interpretation of Transition Probability Plot"),
        html.P("The graph shows the probability density function (PDF) of the particle's position at time t. "
               "The shaded area represents the probability of the particle being within the selected range Δ. "
               "As time increases, the PDF spreads out, indicating that the particle is more likely to be farther from its starting position."),
        dcc.Graph(id='probability-plot'),
        html.H3("Interpretation of Particle Movement Plot"),
        html.P("The graph shows the simulated path of the particle over time. "
               "The red dot represents the particle's position at the selected time t. "
               "This path is a realization of Brownian Motion, where the particle moves randomly due to collisions with surrounding molecules."),
        dcc.Graph(id='particle-movement-plot'),
        html.H3("Definitions and Properties"),
        html.P("1. **Transition Probability**: The likelihood of a particle moving from one position to another over a given time interval. "
               "For Brownian Motion, this is described by a Gaussian (Normal) distribution."),
        html.P("2. **Probability Density Function (PDF)**: A function that describes the likelihood of the particle being at a specific position at a given time. "
               "For Brownian Motion, the PDF is given by: "
               r"\[ p(y) = \frac{1}{\sqrt{2\pi t}} e^{-\frac{y^2}{2t}} \]"),
        html.P("3. **Brownian Motion Properties**: "
               "- **Markov Property**: The future movement of the particle depends only on its current position, not its past. "
               "- **Independent Increments**: The movement of the particle in non-overlapping time intervals is independent. "
               "- **Gaussian Distribution**: The particle's position at any time follows a Gaussian distribution with mean 0 and variance t."),
    ], style={'width': '65%', 'display': 'inline-block'})
])

# Callback to update the plots and probability
@app.callback(
    [Output('probability-plot', 'figure'),
     Output('probability-output', 'children'),
     Output('particle-movement-plot', 'figure')],
    [Input('time-slider', 'value'),
     Input('range-slider', 'value'),
     Input('y-slider', 'value')]
)
def update_plots(t, delta_range, y_position):
    # Generate x values for the plot
    x = np.linspace(-10, 10, 1000)
    
    # Calculate the probability density function (PDF) for Brownian Motion
    mean = 0  # Starting position
    variance = t  # Variance increases with time
    pdf = norm.pdf(x, mean, np.sqrt(variance))
    
    # Calculate the probability of being within the selected range
    lower, upper = delta_range
    probability = norm.cdf(upper, mean, np.sqrt(variance)) - norm.cdf(lower, mean, np.sqrt(variance))
    
    # Calculate the probability density at the selected particle position
    pdf_at_y = norm.pdf(y_position, mean, np.sqrt(variance))
    
    # Create the transition probability plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='PDF'))
    fig1.add_trace(go.Scatter(x=x[(x >= lower) & (x <= upper)], y=pdf[(x >= lower) & (x <= upper)],
                             fill='tozeroy', mode='none', name=f'Probability: {probability:.4f}'))
    fig1.add_trace(go.Scatter(x=[y_position], y=[pdf_at_y], mode='markers', marker=dict(size=10, color='red'), name=f'PDF at y = {y_position:.2f}'))
    fig1.update_layout(
        title=f"Transition Probability Density Function at Time t = {t}",
        xaxis_title="Particle Position (y)",
        yaxis_title="Probability Density",
        showlegend=True
    )
    
    # Create the particle movement plot
    time_values = np.linspace(0, t, 100)
    particle_positions = np.cumsum(np.random.normal(0, np.sqrt(t / 100), 100))  # Simulate Brownian Motion
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time_values, y=particle_positions, mode='lines', name='Particle Path'))
    fig2.add_trace(go.Scatter(x=[t], y=[y_position], mode='markers', marker=dict(size=10, color='red'), name='Selected Position'))
    fig2.update_layout(
        title=f"Particle Movement Over Time (t = 0 to {t})",
        xaxis_title="Time (t)",
        yaxis_title="Particle Position (y)",
        showlegend=True
    )
    
    # Display the probability and PDF at the selected position
    probability_text = f"Probability of particle being in range [{lower}, {upper}] at time t = {t}: {probability:.4f}"
    pdf_text = f"Probability density at particle position y = {y_position:.2f}: {pdf_at_y:.4f}"
    
    return fig1, [html.P(probability_text), html.P(pdf_text)], fig2

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)