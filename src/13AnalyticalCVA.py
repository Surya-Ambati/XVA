import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict

class CapFloorXVA:
    """
    Implements analytic CVA/DVA formulas for interest rate caplets and floorlets
    based on Chapter 5 methodology.
    """
    
    def __init__(self, recovery_rate: float = 0.4):
        self.recovery_rate = recovery_rate
    
    def black_caplet_price(self, forward: float, strike: float, expiry: float, 
                         vol: float, df: float, tau: float, 
                         option_type: str = 'caplet') -> float:
        """
        Black model for caplet/floorlet pricing
        
        Args:
            forward: Forward rate
            strike: Option strike
            expiry: Time to option expiry
            vol: Black volatility
            df: Discount factor to payment date
            tau: Day count fraction for the period
            option_type: 'caplet' or 'floorlet'
        """
        d1 = (np.log(forward/strike) + 0.5 * vol**2 * expiry) / (vol * np.sqrt(expiry))
        d2 = d1 - vol * np.sqrt(expiry)
        
        if option_type.lower() == 'caplet':
            price = df * tau * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
        elif option_type.lower() == 'floorlet':
            price = df * tau * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
        else:
            raise ValueError("Option type must be 'caplet' or 'floorlet'")
            
        return price
    
    def unilateral_cva_caplet(self, caplet: Dict, cds_curve: Dict) -> float:
        """
        Calculate unilateral CVA for a single caplet/floorlet
        
        Args:
            caplet: Dictionary containing:
                - notional: float
                - strike: float
                - fixing_date: float (years)
                - payment_date: float (years)
                - forward_rate: float
                - discount_factor: float
                - daycount_fraction: float
                - volatility: float
                - option_type: 'caplet' or 'floorlet'
                
            cds_curve: Dictionary with:
                - tenors: array-like
                - survival_probs: array-like
                
        Returns:
            float: CVA value
        """
        # Calculate risk-free option value
        option_value = self.black_caplet_price(
            caplet['forward_rate'], 
            caplet['strike'],
            caplet['fixing_date'],
            caplet['volatility'],
            caplet['discount_factor'],
            caplet['daycount_fraction'],
            caplet['option_type']
        )
        
        # Find survival probabilities
        phi_T = np.interp(
            caplet['payment_date'],
            cds_curve['tenors'],
            cds_curve['survival_probs']
        )
        
        # Equation (5.16)
        cva = (1 - self.recovery_rate) * (1 - phi_T) * option_value * caplet['notional']
        
        return cva
    
    def bilateral_cva_caplet(self, caplet: Dict, cpty_cds: Dict, self_cds: Dict) -> float:
        """
        Calculate bilateral CVA for a caplet (equation 5.18)
        
        Args:
            caplet: Dictionary containing caplet details
            cpty_cds: Counterparty CDS curve
            self_cds: Our own CDS curve
            
        Returns:
            float: Bilateral CVA value
        """
        # Calculate risk-free option value
        option_value = self.black_caplet_price(
            caplet['forward_rate'], 
            caplet['strike'],
            caplet['fixing_date'],
            caplet['volatility'],
            caplet['discount_factor'],
            caplet['daycount_fraction'],
            caplet['option_type']
        )
        
        # Time grid for numerical integration (using CDS tenors)
        time_points = cpty_cds['tenors']
        time_points = time_points[time_points <= caplet['payment_date']]
        
        # Initialize CVA
        cva = 0.0
        LGD = 1 - self.recovery_rate
        
        for i in range(len(time_points) - 1):
            t_start = time_points[i]
            t_end = time_points[i+1]
            
            # Survival probability differences
            delta_phi_cpty = (cpty_cds['survival_probs'][i] - 
                             cpty_cds['survival_probs'][i+1])
            
            # Our survival probability at t_start
            phi_self = np.interp(
                t_start,
                self_cds['tenors'],
                self_cds['survival_probs']
            )
            
            # Add to CVA (equation 5.18)
            cva += LGD * option_value * delta_phi_cpty * phi_self * caplet['notional']
        
        return cva
    
    def cva_cap_floor(self, cap_floor: Dict, cds_curve: Dict, 
                     method: str = 'sum') -> float:
        """
        Calculate CVA for a cap/floor (collection of caplets/floorlets)
        
        Args:
            cap_floor: Dictionary containing:
                - notional: float
                - strike: float
                - option_type: 'cap' or 'floor'
                - schedule: List of dictionaries for each period with:
                    - fixing_date
                    - payment_date
                    - forward_rate
                    - discount_factor
                    - daycount_fraction
                    - volatility
                    
            cds_curve: CDS curve information
            method: 'sum' (treat as non-netted) or 'portfolio' (requires MC)
            
        Returns:
            float: CVA value
        """
        if method.lower() == 'sum':
            # Conservative approach - sum CVA of individual caplets/floorlets
            cva = 0.0
            for period in cap_floor['schedule']:
                caplet = {
                    'notional': cap_floor['notional'],
                    'strike': cap_floor['strike'],
                    'option_type': 'caplet' if cap_floor['option_type'] == 'cap' else 'floorlet',
                    **period
                }
                cva += self.unilateral_cva_caplet(caplet, cds_curve)
            return cva
            
        elif method.lower() == 'portfolio':
            raise NotImplementedError(
                "Portfolio method requires Monte Carlo simulation for netting effect")
        else:
            raise ValueError("Method must be 'sum' or 'portfolio'")
        


# Example caplet parameters
caplet = {
    'notional': 1000000,
    'strike': 0.03,
    'fixing_date': 1.0,  # Years
    'payment_date': 1.25,
    'forward_rate': 0.032,
    'discount_factor': 0.97,
    'daycount_fraction': 0.25,
    'volatility': 0.20,
    'option_type': 'caplet'
}

# CDS curve for counterparty
cpty_cds = {
    'tenors': np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5]),
    'survival_probs': np.array([1.0, 0.995, 0.99, 0.985, 0.98, 0.975])
}

# Our own CDS curve for bilateral CVA
self_cds = {
    'tenors': np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5]),
    'survival_probs': np.array([1.0, 0.997, 0.994, 0.991, 0.988, 0.985])
}

# Initialize calculator
calculator = CapFloorXVA(recovery_rate=0.4)

# Calculate unilateral CVA
cva = calculator.unilateral_cva_caplet(caplet, cpty_cds)
print(f"Unilateral CVA: {cva:,.2f}")

# Calculate bilateral CVA
bilateral_cva = calculator.bilateral_cva_caplet(caplet, cpty_cds, self_cds)
print(f"Bilateral CVA: {bilateral_cva:,.2f}")

# Example for a cap (collection of caplets)
cap = {
    'notional': 1000000,
    'strike': 0.03,
    'option_type': 'cap',
    'schedule': [
        {'fixing_date': 0.5, 'payment_date': 0.75, 'forward_rate': 0.031, 
         'discount_factor': 0.98, 'daycount_fraction': 0.25, 'volatility': 0.19},
        {'fixing_date': 1.0, 'payment_date': 1.25, 'forward_rate': 0.032, 
         'discount_factor': 0.97, 'daycount_fraction': 0.25, 'volatility': 0.20},
        {'fixing_date': 1.5, 'payment_date': 1.75, 'forward_rate': 0.033, 
         'discount_factor': 0.96, 'daycount_fraction': 0.25, 'volatility': 0.21}
    ]
}

cap_cva = calculator.cva_cap_floor(cap, cpty_cds, method='sum')
print(f"Cap CVA (conservative): {cap_cva:,.2f}")