# Import required libraries
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, norm
import plotly.graph_objects as go
from datetime import datetime

# 1. Initialize sample data
def generate_sample_data(num_points=100):
    """Generate synthetic exposure and financial data for demo"""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.today(), periods=num_points)
    
    # Simulate exposure paths (MTM values)
    base_exposure = np.random.normal(1_000_000, 250_000)
    exposures = base_exposure + np.cumsum(np.random.normal(0, 50_000, num_points))
    
    # Simulate financial ratios (leverage, coverage, etc.)
    data = pd.DataFrame({
        'date': dates,
        'exposure': np.maximum(exposures, 0),
        'leverage_ratio': np.clip(np.random.normal(2.0, 0.5, num_points), 1.0, 4.0),
        'interest_coverage': np.clip(np.random.normal(5.0, 2.0, num_points), 1.0, 15.0),
        'current_ratio': np.clip(np.random.normal(1.8, 0.3, num_points), 1.0, 3.0),
        'profit_margin': np.clip(np.random.normal(0.12, 0.05, num_points), 0.01, 0.25),
        'equity_volatility': np.clip(np.random.normal(0.35, 0.1, num_points), 0.15, 0.6),
        'debt_to_assets': np.clip(np.random.normal(0.65, 0.15, num_points), 0.3, 0.9),
        'credit_rating': np.random.choice(['AA', 'A', 'BBB', 'BB'], num_points, p=[0.1, 0.3, 0.4, 0.2])
    })
    
    # Create some WWR/RWR relationships
    data['leverage_ratio'] = data['leverage_ratio'] + 0.000001 * data['exposure']  # WWR
    data['interest_coverage'] = data['interest_coverage'] - 0.000002 * data['exposure']  # RWR
    
    return data.set_index('date')

# 2. Define credit proxy models
class CreditProxyModels:
    @staticmethod
    def financial_ratio_proxy(data):
        """Calculate credit proxy using financial ratios"""
        return (
            0.4 * data['leverage_ratio'] +
            0.3 * (1 / data['interest_coverage']) +  # Inverse because higher coverage is better
            0.2 * (1 / data['current_ratio']) +
            0.1 * (1 / data['profit_margin'])
        )
    
    @staticmethod
    def equity_credit_model(data):
        """Merton model inspired equity-to-credit transformation"""
        dd = (np.log(1/data['debt_to_assets']) + (0.03 + 0.5*data['equity_volatility']**2)) / data['equity_volatility']
        pd = norm.cdf(-dd)  # Probability of default
        return pd * 0.6 * 10000  # Convert to bps with 60% LGD
    
    @staticmethod
    def rating_based_spread(data):
        """Convert ratings to spread equivalents"""
        rating_map = {'AA': 50, 'A': 80, 'BBB': 120, 'BB': 200}
        return data['credit_rating'].map(rating_map)

# 3. WWR Calculator Implementation
class WWRAnalyzer:
    def __init__(self, data):
        self.data = data
        self.proxy_models = CreditProxyModels()
        self.results = {}
    
    def calculate_all_proxies(self):
        """Run all available proxy methods"""
        self.results['financial_ratios'] = self._calculate_wwr('financial_ratio_proxy')
        self.results['equity_model'] = self._calculate_wwr('equity_credit_model')
        self.results['rating_based'] = self._calculate_wwr('rating_based_spread')
        return self.results
    
    def _calculate_wwr(self, method_name):
        """Calculate WWR using specified proxy method"""
        proxy_method = getattr(self.proxy_models, method_name)
        credit_metric = proxy_method(self.data)
        
        # Calculate correlation with exposure
        valid_data = pd.DataFrame({
            'exposure': self.data['exposure'],
            'credit_metric': credit_metric
        }).dropna()
        
        if len(valid_data) < 10:  # Minimum data points requirement
            return {'error': 'Insufficient data'}
        
        corr, pval = pearsonr(valid_data['exposure'], valid_data['credit_metric'])
        
        return {
            'correlation': corr,
            'p_value': pval,
            'data_points': len(valid_data),
            'method': method_name,
            'credit_metric': credit_metric
        }
    
    def weighted_wwr_estimate(self):
        """Calculate weighted average of all proxy methods"""
        weights = {'financial_ratios': 0.5, 'equity_model': 0.3, 'rating_based': 0.2}
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_results:
            return {'error': 'No valid methods available'}
        
        weighted_corr = sum(
            result['correlation'] * weights[method]
            for method, result in valid_results.items()
        )
        
        return {
            'weighted_correlation': weighted_corr,
            'methods_used': list(valid_results.keys()),
            'method_details': valid_results
        }
    
    def visualize_results(self):
        """Create interactive visualization of results"""
        fig = go.Figure()
        
        # Add correlation bars
        methods = []
        correlations = []
        for method, result in self.results.items():
            if 'error' not in result:
                methods.append(method)
                correlations.append(result['correlation'])
        
        fig.add_trace(go.Bar(
            x=methods,
            y=correlations,
            name='Correlation',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
        ))
        
        # Add weighted average line
        weighted = self.weighted_wwr_estimate()
        if 'error' not in weighted:
            fig.add_hline(
                y=weighted['weighted_correlation'],
                line_dash="dot",
                annotation_text=f"Weighted Average: {weighted['weighted_correlation']:.2f}",
                line_color="red"
            )
        
        fig.update_layout(
            title='WWR Correlation Across Different Proxy Methods',
            yaxis=dict(title='Correlation Coefficient', range=[-1, 1]),
            hovermode='x unified'
        )
        
        return fig
    
    def plot_exposure_vs_proxy(self, method_name):
        """Scatter plot of exposure vs credit metric for a specific method"""
        if method_name not in self.results or 'error' in self.results[method_name]:
            return go.Figure()
        
        result = self.results[method_name]
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.data['exposure'],
            y=result['credit_metric'],
            mode='markers',
            name=f'{method_name} vs Exposure'
        ))
        
        # Add trendline
        z = np.polyfit(self.data['exposure'], result['credit_metric'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=self.data['exposure'],
            y=p(self.data['exposure']),
            name='Trendline',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f'Exposure vs {method_name} Credit Proxy',
            xaxis_title='Exposure (MTM)',
            yaxis_title='Credit Proxy Value'
        )
        
        return fig

# 4. Example Usage
if __name__ == "__main__":
    print("=== WWR Analysis Without CDS Data ===")
    
    # Step 1: Generate or load your data
    sample_data = generate_sample_data(100)
    print("\nSample data preview:")
    print(sample_data.head())
    
    # Step 2: Initialize analyzer
    analyzer = WWRAnalyzer(sample_data)
    
    # Step 3: Calculate all proxy methods
    print("\nCalculating WWR using all proxy methods...")
    results = analyzer.calculate_all_proxies()
    
    print("\nIndividual method results:")
    for method, result in results.items():
        if 'error' in result:
            print(f"{method}: {result['error']}")
        else:
            print(f"{method}: Correlation = {result['correlation']:.2f}, p-value = {result['p_value']:.4f}")
    
    # Step 4: Get weighted consensus
    weighted_result = analyzer.weighted_wwr_estimate()
    print("\nWeighted WWR estimate:")
    print(f"Correlation: {weighted_result['weighted_correlation']:.2f}")
    print(f"Methods used: {', '.join(weighted_result['methods_used'])}")
    
    # Step 5: Visualize results
    print("\nGenerating visualizations...")
    
    # # Show correlation comparison
    # corr_fig = analyzer.visualize_results()
    # corr_fig.show()
    
    # # Show exposure vs proxy relationships
    # for method in results.keys():
    #     scatter_fig = analyzer.plot_exposure_vs_proxy(method)
    #     scatter_fig.show()
    
    print("\nAnalysis complete!")