import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from datetime import datetime, timedelta
import scipy.stats as stats
import plotly.graph_objects as go
import warnings

class WWRCalculator:
    def __init__(self):
        self.mtm_results = None
        self.credit_history = None
        self.wwr_correlations = None
        self.cva_results = None
        
    def generate_dummy_data(self, num_dates=90, num_counterparties=8):
        """Generate dummy market and credit data with both WWR and RWR examples"""
        dates = [datetime.today() - timedelta(days=x) for x in range(num_dates)]
        counterparties = [f"CP_{i}" for i in range(num_counterparties)]
        
        # Market data (MTM values) with some autocorrelation
        np.random.seed(42)
        base_values = np.random.normal(1_000_000, 250_000, size=num_counterparties)
        mtm_data = {}
        
        for date in dates:
            daily_move = np.random.normal(0, 100_000, size=num_counterparties)
            mtm_data[date] = {
                cp: max(base_values[i] + daily_move[i], 0) 
                for i, cp in enumerate(counterparties)
            }
            base_values += daily_move * 0.2
        
        # Credit data (CDS spreads) with deliberate WWR and RWR relationships
        credit_data = {}
        for date in dates:
            credit_data[date] = {}
            for i, cp in enumerate(counterparties):
                exposure = mtm_data[date][cp]
                base_spread = 50 + np.random.normal(0, 10)
                
                # Create different risk profiles
                if i < 3:  # First 3 counterparties: Strong WWR
                    exposure_component = 0.0002 * exposure  # 2bp per $100k exposure
                elif i < 5:  # Next 2: Mild WWR
                    exposure_component = 0.00005 * exposure  # 0.5bp per $100k
                elif i < 7:  # Next 2: Mild RWR
                    exposure_component = -0.00005 * exposure  # -0.5bp per $100k
                else:  # Last one: Strong RWR
                    exposure_component = -0.00015 * exposure  # -1.5bp per $100k
                
                credit_data[date][cp] = max(base_spread + exposure_component, 10)
        
        self.mtm_results = pd.DataFrame.from_dict(mtm_data, orient='index')
        self.credit_history = pd.DataFrame.from_dict(credit_data, orient='index')
        
    def calculate_correlations(self):
        """Calculate WWR correlations and basic CVA adjustment"""
        if self.mtm_results is None or self.credit_history is None:
            raise ValueError("Data not loaded. Call generate_dummy_data() first.")
            
        correlations = {}
        p_values = {}
        regression_params = {}
        cva_adjustments = {}
        
        # Base CVA parameters
        lgd = 0.6  # Loss given default (60%)
        discount_rate = 0.05  # Annual discount rate
        
        for cp in self.mtm_results.columns:
            # Get exposure and credit spread data
            exposures = self.mtm_results[cp].values
            spreads = self.credit_history[cp].values
            
            # Skip if constant values (undefined correlation)
            if np.all(exposures == exposures[0]) or np.all(spreads == spreads[0]):
                correlations[cp] = 0
                p_values[cp] = 1
                slope = 0
                intercept = np.mean(spreads)
                cva_factor = 1.0
            else:
                # Calculate Pearson correlation and p-value
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr, pval = stats.pearsonr(exposures, spreads)
                
                # Regression parameters
                slope, intercept = np.polyfit(exposures, spreads, 1)
                
                # Calculate CVA adjustment factor based on correlation
                if corr > 0.3 and pval < 0.05:  # Significant WWR
                    cva_factor = 1 + corr  # Increase CVA
                elif corr < -0.3 and pval < 0.05:  # Significant RWR
                    cva_factor = 1 - abs(corr)  # Decrease CVA
                else:
                    cva_factor = 1.0  # No adjustment
            
            correlations[cp] = corr if 'corr' in locals() else 0
            p_values[cp] = pval if 'pval' in locals() else 1
            regression_params[cp] = {'slope': slope, 'intercept': intercept}
            cva_adjustments[cp] = cva_factor
            
        self.wwr_correlations = pd.DataFrame({
            'Counterparty': correlations.keys(),
            'Correlation': correlations.values(),
            'P-Value': p_values.values(),
            'Slope': [x['slope'] for x in regression_params.values()],
            'Intercept': [x['intercept'] for x in regression_params.values()],
            'CVA_Factor': cva_adjustments.values()
        })
        
        # Calculate sample CVA values (simplified for demonstration)
        avg_exposures = self.mtm_results.mean()
        avg_spreads = self.credit_history.mean() / 10000  # Convert bps to decimal
        days = len(self.mtm_results)
        
        # Create CVA results with proper alignment
        cva_data = []
        for cp in avg_exposures.index:
            base_cva = avg_exposures[cp] * avg_spreads[cp] * lgd * (days/365) * np.exp(-discount_rate * (days/365))
            adj_factor = self.wwr_correlations.loc[self.wwr_correlations['Counterparty'] == cp, 'CVA_Factor'].values[0]
            adj_cva = base_cva * adj_factor
            
            cva_data.append({
                'Counterparty': cp,
                'Base_CVA': base_cva,
                'Adjusted_CVA': adj_cva,
                'Adjustment_Factor': adj_factor
            })
        
        self.cva_results = pd.DataFrame(cva_data)
        
    def visualize_results(self):
        """Create interactive Dash visualization with Data Explorer tab"""
        if self.wwr_correlations is None:
            raise ValueError("No correlation results. Call calculate_correlations() first.")
            
        app = dash.Dash(__name__)
        
        # Create the initial figures
        corr_fig = px.bar(
            self.wwr_correlations,
            x='Counterparty',
            y='Correlation',
            title='WWR Correlations by Counterparty',
            color='Correlation',
            color_continuous_scale='RdYlGn',
            range_color=[-1, 1],
            hover_data=['P-Value', 'Slope', 'CVA_Factor']
        ).update_layout(showlegend=False)
        
        cva_fig = px.bar(
            self.cva_results,
            x='Counterparty',
            y=['Base_CVA', 'Adjusted_CVA'],
            title='CVA Before and After WWR Adjustment',
            barmode='group',
            labels={'value': 'CVA Amount', 'variable': 'CVA Type'}
        )
        
        # Initial Data Explorer figure
        initial_data_type = 'exposures'
        explorer_fig = go.Figure()
        if initial_data_type == 'exposures':
            data = self.mtm_results
            title = "Market Exposures Over Time"
            y_label = "Exposure (USD)"
        else:
            data = self.credit_history
            title = "Credit Spreads Over Time"
            y_label = "CDS Spread (bps)"
            
        for cp in data.columns:
            explorer_fig.add_trace(go.Scatter(
                x=data.index,
                y=data[cp],
                mode='lines+markers',
                name=cp
            ))
        
        explorer_fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=y_label,
            hovermode='x unified'
        )
        
        app.layout = html.Div([
            html.H1("Wrong-Way Risk Analysis Dashboard", style={'textAlign': 'center'}),
            
            dcc.Tabs([
                dcc.Tab(label='Risk Correlation', children=[
                    html.Div([
                        dcc.Graph(id='corr-bar', figure=corr_fig),
                        html.Div(id='overview-interpretation')
                    ], style={'padding': '20px'})
                ]),
                
                dcc.Tab(label='Exposure Analysis', children=[
                    html.Div([
                        dcc.Dropdown(
                            id='cp-dropdown',
                            options=[{'label': cp, 'value': cp} for cp in self.mtm_results.columns],
                            value=self.mtm_results.columns[0],
                            style={'width': '50%', 'margin': '10px'}
                        ),
                        dcc.Graph(id='scatter-plot'),
                        html.Div(id='scatter-interpretation')
                    ], style={'padding': '20px'})
                ]),
                
                dcc.Tab(label='CVA Impact', children=[
                    html.Div([
                        dcc.Graph(id='cva-plot', figure=cva_fig),
                        html.Div(id='cva-interpretation')
                    ], style={'padding': '20px'})
                ]),
                
                dcc.Tab(label='Data Explorer', children=[
                    html.Div([
                        dcc.Dropdown(
                            id='data-type-dropdown',
                            options=[
                                {'label': 'Market Exposures', 'value': 'exposures'},
                                {'label': 'Credit Spreads', 'value': 'spreads'}
                            ],
                            value='exposures',
                            style={'width': '50%', 'margin': '10px'}
                        ),
                        dcc.Graph(id='data-explorer-plot', figure=explorer_fig)
                    ], style={'padding': '20px'})
                ])
            ]),
            
            html.Div([
                html.H3("Methodology"),
                html.Ul([
                    html.Li(children=[
                        html.Span("Red bars: ", style={'color': 'red'}),
                        "Wrong-Way Risk (positive correlation between exposure and credit risk)"
                    ]),
                    html.Li(children=[
                        html.Span("Green bars: ", style={'color': 'green'}),
                        "Right-Way Risk (negative correlation)"
                    ]),
                    html.Li("CVA adjusted by correlation strength (factors shown in hover data)"),
                    html.Li("P-values < 0.05 indicate statistically significant relationships")
                ])
            ], style={
                'margin': '20px',
                'padding': '15px',
                'borderTop': '1px solid #eee',
                'backgroundColor': '#f5f5f5'
            })
        ])
        
        @app.callback(
            Output('scatter-plot', 'figure'),
            Input('cp-dropdown', 'value')
        )
        def update_scatter(selected_cp):
            x = self.mtm_results[selected_cp]
            y = self.credit_history[selected_cp]
            
            fig = px.scatter(
                x=x,
                y=y,
                title=f'Exposure vs Credit Spread for {selected_cp}',
                labels={'x': 'Exposure (MTM)', 'y': 'CDS Spread (bps)'},
                trendline='ols'
            )
            
            fig.add_hline(y=y.mean(), line_dash="dot", 
                         annotation_text=f"Avg Spread: {y.mean():.1f}bps",
                         annotation_position="bottom right")
            fig.add_vline(x=x.mean(), line_dash="dot", 
                         annotation_text=f"Avg Exposure: ${x.mean():,.0f}",
                         annotation_position="top right")
            
            return fig
            
        @app.callback(
            Output('scatter-interpretation', 'children'),
            Input('cp-dropdown', 'value')
        )
        def update_scatter_interpretation(selected_cp):
            if selected_cp is None:
                return ""
                
            corr_row = self.wwr_correlations[self.wwr_correlations['Counterparty'] == selected_cp]
            cva_row = self.cva_results[self.cva_results['Counterparty'] == selected_cp]
            
            corr = corr_row['Correlation'].values[0]
            pval = corr_row['P-Value'].values[0]
            slope = corr_row['Slope'].values[0]
            cva_adj = cva_row['Adjustment_Factor'].values[0]
            base_cva = cva_row['Base_CVA'].values[0]
            adj_cva = cva_row['Adjusted_CVA'].values[0]
            
            risk_type = "Wrong-Way Risk" if corr > 0 else "Right-Way Risk"
            significance = "significant" if pval < 0.05 else "not significant"
            
            return html.Div([
                html.H4(f"{selected_cp} Risk Analysis"),
                html.P(f"Correlation: {corr:.2f} ({risk_type}, {significance})"),
                html.P(f"P-value: {pval:.4f}"),
                html.P(f"Slope: {slope:.6f} (CDS spread changes by {slope*100000:.1f}bps per $100k exposure)"),
                html.P(f"CVA Adjustment: {cva_adj:.2f}x"),
                html.P(f"Base CVA: ${base_cva:,.2f} → Adjusted CVA: ${adj_cva:,.2f}"),
                html.P("Recommended Action:"),
                html.Ul([
                    html.Li("Increase collateral requirements" if corr > 0.3 else 
                          "Review hedging strategy" if corr > 0.1 else
                          "Monitor relationship" if corr > -0.1 else
                          "May reduce collateral requirements")
                ])
            ])
            
        @app.callback(
            Output('overview-interpretation', 'children'),
            Input('corr-bar', 'hoverData')
        )
        def update_overview_interpretation(hover_data):
            if hover_data is None:
                return html.Div("Hover over bars to see details")
                
            point = hover_data['points'][0]
            cp = point['x']
            corr = point['y']
            
            # Look up the p-value directly from wwr_correlations
            corr_row = self.wwr_correlations[self.wwr_correlations['Counterparty'] == cp]
            if corr_row.empty:
                return html.Div(f"No data found for {cp}")
            pval = corr_row['P-Value'].values[0]
            
            # Determine the risk type and significance
            if corr > 0.3 and pval < 0.05:
                risk_message = "Significant Wrong-Way Risk (WWR)"
            elif corr < -0.3 and pval < 0.05:
                risk_message = "Significant Right-Way Risk (RWR)"
            else:
                risk_message = "No significant relationship"
            
            return html.Div([
                html.H4(f"{cp} Summary"),
                html.P(f"Correlation: {corr:.2f}"),
                html.P(f"P-value: {pval:.4f}"),
                html.P(risk_message)
            ])
            
        @app.callback(
            Output('cva-interpretation', 'children'),
            Input('cva-plot', 'hoverData')
        )
        def update_cva_interpretation(hover_data):
            if hover_data is None:
                return html.Div("Hover over bars to see CVA details")
                
            point = hover_data['points'][0]
            cp = point['x']
            cva_type = point['legendgroup']
            cva_value = point['y']
            
            cva_row = self.cva_results[self.cva_results['Counterparty'] == cp]
            base_cva = cva_row['Base_CVA'].values[0]
            adj_cva = cva_row['Adjusted_CVA'].values[0]
            adj_factor = cva_row['Adjustment_Factor'].values[0]
            
            return html.Div([
                html.H4(f"{cp} CVA Analysis"),
                html.P(f"Base CVA: ${base_cva:,.2f}"),
                html.P(f"Adjusted CVA: ${adj_cva:,.2f}"),
                html.P(f"Adjustment Factor: {adj_factor:.2f}x"),
                html.P(f"Current selection: {cva_type} = ${cva_value:,.2f}")
            ])
            
        @app.callback(
            Output('data-explorer-plot', 'figure'),
            Input('data-type-dropdown', 'value')
        )
        def update_data_explorer(selected_data_type):
            if selected_data_type == 'exposures':
                data = self.mtm_results
                title = "Market Exposures Over Time"
                y_label = "Exposure (USD)"
            else:
                data = self.credit_history
                title = "Credit Spreads Over Time"
                y_label = "CDS Spread (bps)"
            
            fig = go.Figure()
            
            for cp in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[cp],
                    mode='lines+markers',
                    name=cp
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title=y_label,
                hovermode='x unified'
            )
            
            return fig
            
        app.run(debug=True, port=8050)

if __name__ == "__main__":
    print("Initializing WWR Analysis...")
    wwr = WWRCalculator()
    
    print("Generating dummy data with WWR and RWR examples...")
    wwr.generate_dummy_data()
    
    print("Calculating correlations and CVA adjustments...")
    wwr.calculate_correlations()
    
    print("\nCorrelation Results:")
    print(wwr.wwr_correlations[['Counterparty', 'Correlation', 'P-Value', 'CVA_Factor']])
    
    print("\nCVA Results:")
    print(wwr.cva_results)
    
    print("\nLaunching dashboard...")
    wwr.visualize_results()