import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple

class FXForwardXVA:
    """
    Implements analytic CVA/DVA formulas for FX Forwards
    based on Garman-Kohlhagen model for FX options.
    """
    
    def __init__(self, recovery_rate: float = 0.4):
        self.recovery_rate = recovery_rate
    
    def gk_fx_option(self, spot: float, strike: float, rd: float, rf: float, 
                    T: float, vol: float, option_type: str) -> float:
        """
        Garman-Kohlhagen model for FX options pricing
        
        Args:
            spot: Current FX spot rate
            strike: Option strike
            rd: Domestic interest rate
            rf: Foreign interest rate
            T: Time to maturity
            vol: FX volatility
            option_type: 'call' or 'put'
        """
        d1 = (np.log(spot/strike) + (rd - rf + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = spot * np.exp(-rf*T) * norm.cdf(d1) - strike * np.exp(-rd*T) * norm.cdf(d2)
        elif option_type.lower() == 'put':
            price = strike * np.exp(-rd*T) * norm.cdf(-d2) - spot * np.exp(-rf*T) * norm.cdf(-d1)
        else:
            raise ValueError("Option type must be 'call' or 'put'")
            
        return price
    
    def unilateral_cva_fx_forward(self, fx_forward: Dict, cds_curve: Dict) -> float:
        """
        Calculate unilateral CVA for FX Forward
        
        Args:
            fx_forward: Dictionary containing:
                - notional: float
                - strike: float
                - maturity: float (years)
                - spot_rate: float
                - dom_rate: float (domestic interest rate)
                - for_rate: float (foreign interest rate)
                - volatility: float
                - direction: 'long' or 'short' (domestic currency)
                
            cds_curve: Dictionary with:
                - tenors: array-like
                - survival_probs: array-like
                
        Returns:
            float: CVA value
        """
        # Determine option type based on forward direction
        if fx_forward['direction'].lower() == 'long':
            option_type = 'call'  # We're long domestic currency -> care about upside
        else:
            option_type = 'put'   # We're short domestic currency -> care about downside
        
        # Calculate FX option value
        option_value = self.gk_fx_option(
            fx_forward['spot_rate'],
            fx_forward['strike'],
            fx_forward['dom_rate'],
            fx_forward['for_rate'],
            fx_forward['maturity'],
            fx_forward['volatility'],
            option_type
        )
        
        # Find survival probability at maturity
        phi_T = np.interp(
            fx_forward['maturity'],
            cds_curve['tenors'],
            cds_curve['survival_probs']
        )
        
        # Equation (5.21)
        cva = (1 - self.recovery_rate) * (1 - phi_T) * option_value * fx_forward['notional']
        
        return cva
    
    def bilateral_cva_fx_forward(self, fx_forward: Dict, cpty_cds: Dict, 
                               self_cds: Dict) -> Tuple[float, float]:
        """
        Calculate bilateral CVA/DVA for FX Forward
        
        Args:
            fx_forward: Dictionary containing FX forward details
            cpty_cds: Counterparty CDS curve
            self_cds: Our own CDS curve
            
        Returns:
            Tuple of (CVA, DVA)
        """
        # Calculate FX call and put values
        call_value = self.gk_fx_option(
            fx_forward['spot_rate'],
            fx_forward['strike'],
            fx_forward['dom_rate'],
            fx_forward['for_rate'],
            fx_forward['maturity'],
            fx_forward['volatility'],
            'call'
        )
        
        put_value = self.gk_fx_option(
            fx_forward['spot_rate'],
            fx_forward['strike'],
            fx_forward['dom_rate'],
            fx_forward['for_rate'],
            fx_forward['maturity'],
            fx_forward['volatility'],
            'put'
        )
        
        # Time grid for numerical integration
        time_points = cpty_cds['tenors']
        time_points = time_points[time_points <= fx_forward['maturity']]
        
        # Initialize CVA and DVA
        cva = 0.0
        dva = 0.0
        LGD_cpty = 1 - self.recovery_rate
        LGD_self = 1 - self.recovery_rate
        
        for i in range(len(time_points) - 1):
            t_start = time_points[i]
            t_end = time_points[i+1]
            
            # Survival probability differences
            delta_phi_cpty = (cpty_cds['survival_probs'][i] - 
                             cpty_cds['survival_probs'][i+1])
            delta_phi_self = (self_cds['survival_probs'][i] - 
                             self_cds['survival_probs'][i+1])
            
            # Joint survival probabilities
            phi_cpty = cpty_cds['survival_probs'][i]
            phi_self = self_cds['survival_probs'][i]
            
            # Add to CVA and DVA (equation 5.22)
            cva += LGD_cpty * delta_phi_cpty * phi_self * call_value * fx_forward['notional']
            dva += LGD_self * delta_phi_self * phi_cpty * put_value * fx_forward['notional']
        
        return cva, dva
    

# Example FX Forward parameters
fx_forward = {
    'notional': 1000000,  # Domestic currency amount
    'strike': 1.20,       # FX rate
    'maturity': 1.0,      # Years
    'spot_rate': 1.18,
    'dom_rate': 0.05,     # Domestic interest rate
    'for_rate': 0.02,     # Foreign interest rate
    'volatility': 0.15,   # FX volatility
    'direction': 'long'   # Long domestic currency
}

# CDS curves
cpty_cds = {
    'tenors': np.array([0.25, 0.5, 0.75, 1.0, 1.25]),
    'survival_probs': np.array([0.99, 0.98, 0.97, 0.96, 0.95])
}

self_cds = {
    'tenors': np.array([0.25, 0.5, 0.75, 1.0, 1.25]),
    'survival_probs': np.array([0.995, 0.99, 0.985, 0.98, 0.975])
}

# Initialize calculator
calculator = FXForwardXVA(recovery_rate=0.4)

# Unilateral CVA
cva = calculator.unilateral_cva_fx_forward(fx_forward, cpty_cds)
print(f"Unilateral CVA: {cva:,.2f}")

# Bilateral CVA/DVA
cva, dva = calculator.bilateral_cva_fx_forward(fx_forward, cpty_cds, self_cds)
print(f"Bilateral CVA: {cva:,.2f}, DVA: {dva:,.2f}")