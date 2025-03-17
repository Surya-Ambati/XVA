import numpy as np
import matplotlib.pyplot as plt

""" simulate_brownian_motion: 
Filtrations and Brownian Motion

Concept Definition:
Filtration: Represents the information available up to time t. 
It defines what is known and what remains uncertain.

Brownian Motion: A stochastic process with independent, normally distributed increments.
"""

def simulate_brownian_motion(T=1, N=100, num_paths=5):
    """
    Simulate Brownian motion paths.
    
    Parameters:
    - T (float): Total time (default=1).
    - N (int): Number of time steps (default=100).
    - num_paths (int): Number of simulated paths (default=5).
    
    Returns:
    - t (array): Time grid.
    - W (array): Simulated Brownian motion paths (shape: num_paths x N).
    """
    dt = T / N
    t = np.linspace(0, T, N)
    dW = np.random.normal(0, np.sqrt(dt), (num_paths, N))
    W = np.cumsum(dW, axis=1)
    return t, W

# Generate paths
t, W = simulate_brownian_motion(T=1, N=100, num_paths=5)

"""
Graph Interpretation:
Each path represents a realization of Brownian motion. The spread of paths widens over time, 
illustrating increasing uncertainty (volatility scales with squareroot of t ).
"""

# Plot
plt.figure(figsize=(10, 6))
for i in range(W.shape[0]):
    plt.plot(t, W[i], label=f'Path {i+1}')
plt.title("Brownian Motion Paths (Filtration Visualization)")
plt.xlabel("Time (t)")
plt.ylabel("W(t)")
plt.legend()
plt.grid(True)
plt.show()





""" nested_monte_carlo: 
Conditional Expectations (Tower Property)
Concept Definition:
Conditional Expectation: E[X∣Ft] is the expected value of X given information up to time t.

Tower Property: 
E[E[X∣Ft]∣Fs] = E[X∣Fs] for s≤t.
"""


def nested_monte_carlo(T=1, s=0.5, N_outer=1000, N_inner=100):
    """
    Compute conditional expectation of Brownian motion at time T given information at time s.
    
    Parameters:
    - T (float): Total time (default=1).
    - s (float): Intermediate time (default=0.5).
    - N_outer (int): Number of outer paths (default=1000).
    - N_inner (int): Number of inner paths for nested simulation (default=100).
    
    Returns:
    - W_s (array): Brownian motion values at time s.
    - E_Wt_given_Fs (array): Conditional expectations.
    """
    # Simulate outer paths up to time s
    _, W_outer = simulate_brownian_motion(T=s, N=int(s*100), num_paths=N_outer)
    W_s = W_outer[:, -1]  # Values at time s
    
    # Nested simulation for E[W_T | F_s]
    E_Wt_given_Fs = []
    for w in W_s:
        # Simulate from s to T
        _, W_inner = simulate_brownian_motion(T=T-s, N=int((T-s)*100), num_paths=N_inner)
        W_t = w + W_inner[:, -1]  # W_T = W_s + (W_T - W_s)
        E_Wt_given_Fs.append(np.mean(W_t))
    
    return W_s, np.array(E_Wt_given_Fs)

# Run simulation
W_s, E_Wt_given_Fs = nested_monte_carlo(T=1, s=0.5, N_outer=1000, N_inner=100)

"""
Graph Interpretation:
The red line Ws represents the theoretical result E[WT∣Fs]=Ws . 
The scatter plot shows simulated results, confirming the martingale property of Brownian motion.
"""

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(W_s, E_Wt_given_Fs, alpha=0.5, label='E[W_T | F_s]')
plt.plot(W_s, W_s, color='red', label='W_s (Theory)')
plt.title("Conditional Expectation of Brownian Motion (Tower Property)")
plt.xlabel("W_s (Value at s=0.5)")
plt.ylabel("E[W_T | F_s]")
plt.legend()
plt.grid(True)
plt.show()





def girsanov_transform(T=1, N=100, mu_P=0.2, mu_Q=0.05):
    """
    Measure Changes (Girsanov Theorem)
    Concept Definition:
    Measure Change: Adjusts the drift of a stochastic process under a new probability measure (e.g., risk-neutral measure Q).

    Simulate Brownian motion under real-world (P) and risk-neutral (Q) measures.
    
    Parameters:
    - T (float): Total time (default=1).
    - N (int): Number of time steps (default=100).
    - mu_P (float): Drift under P (real-world measure).
    - mu_Q (float): Drift under Q (risk-neutral measure).
    
    Returns:
    - t (array): Time grid.
    - S_P (array): Process under P.
    - S_Q (array): Process under Q.
    """
    dt = T / N
    t = np.linspace(0, T, N)
    dW = np.random.normal(0, np.sqrt(dt), N)
    
    # Real-world measure P
    S_P = np.cumsum(mu_P * dt + dW)
    
    # Risk-neutral measure Q (drift adjustment)
    S_Q = np.cumsum(mu_Q * dt + dW)
    
    return t, S_P, S_Q

# Simulate paths
t, S_P, S_Q = girsanov_transform(T=1, N=100, mu_P=0.2, mu_Q=0.05)



"""
Graph Interpretation:
The process under Q has a reduced drift ( μQ = 0.05) compared to 
P (μP = 0.2), reflecting the risk-neutral adjustment.
"""

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, S_P, label='Real-World Measure (P)')
plt.plot(t, S_Q, label='Risk-Neutral Measure (Q)')
plt.title("Measure Change via Girsanov Theorem")
plt.xlabel("Time (t)")
plt.ylabel("Process Value")
plt.legend()
plt.grid(True)
plt.show()





def check_martingale(num_simulations=1000):
    """
    Verify if Brownian motion is a martingale.
    
    Parameters:
    - num_simulations (int): Number of simulated paths (default=1000).
    """
    t, W = simulate_brownian_motion(T=1, N=100, num_paths=num_simulations)
    E_Wt = np.mean(W[:, -1])  # E[W_T]
    
    # Plot Brownian motion paths
    plt.figure(figsize=(10, 6))
    for i in range(W.shape[0]):
        plt.plot(t, W[i], alpha=0.1, color='blue')
    plt.title(f"Brownian Motion Paths (E[W_T] = {E_Wt:.4f})")
    plt.xlabel("Time (t)")
    plt.ylabel("W(t)")
    plt.grid(True)
    plt.show()

# Run simulation
check_martingale(num_simulations=1000)

# Notes:

# Objective:
# Verify that Brownian motion is a martingale by checking if the expected value of WTis zero.

# Martingale Graph Interpretation:
# The graph shows multiple Brownian motion paths. The expected value of W T (the value at t=1) is close to zero, 
# confirming the martingale property of Brownian motion.

# The spread of paths increases over time, reflecting the increasing uncertainty (volatility scales with 
# square root of t).


def nested_option_pricing(S0=100, K=110, r=0.05, T=1, s=0.5, N_outer=1000, N_inner=100):
    """
    Price a European call option with nested Monte Carlo.
    
    Parameters:
    - S0 (float): Initial stock price (default=100).
    - K (float): Strike price (default=110).
    - r (float): Risk-free rate (default=0.05).
    - T (float): Time to maturity (default=1).
    - s (float): Intermediate conditioning time (default=0.5).
    - N_outer (int): Number of outer paths (default=1000).
    - N_inner (int): Number of inner paths (default=100).
    """
    # Simulate stock paths up to s
    t, S = simulate_brownian_motion(T=s, N=int(s*100), num_paths=N_outer)
    S_s = S[:, -1]  # Stock price at s
    
    # Nested simulation for E[(S_T - K)^+ | F_s]
    payoffs = []
    for s_val in S_s:
        _, W_inner = simulate_brownian_motion(T=T-s, N=int((T-s)*100), num_paths=N_inner)
        S_T = s_val * np.exp((r - 0.5*0.2**2)*(T-s) + 0.2*np.cumsum(W_inner, axis=1)[:, -1])
        payoff = np.maximum(S_T - K, 0)
        payoffs.append(np.mean(payoff) * np.exp(-r*(T-s)))
    
    option_price = np.mean(payoffs)
    print(f"Option Price = {option_price:.2f}")
    
    # Plot stock price paths up to s
    plt.figure(figsize=(10, 6))
    for i in range(S.shape[0]):
        plt.plot(t, S[i], alpha=0.1, color='blue')
    plt.title(f"Stock Price Paths up to s={s} (Option Price = {option_price:.2f})")
    plt.xlabel("Time (t)")
    plt.ylabel("Stock Price")
    plt.grid(True)
    plt.show()

# Run simulation
nested_option_pricing(S0=100, K=110, r=0.05, T=1, s=0.5)


# Notes:

# Objective:
# Price a European call option using nested Monte Carlo simulations. The payoff is max⁡(ST − K, 0), where 
# ST is the stock price at maturity T.

# nested_option_pricing Graph Interpretation:
# The graph shows simulated stock price paths up to time s=0.5. 
# The option price is calculated using nested Monte Carlo simulations, where each path at s=0.5 
# is used to simulate future stock prices up to T=1.

# The option price reflects the expected payoff max(ST − K,0), discounted to the present value.
