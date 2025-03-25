import numpy as np
import matplotlib.pyplot as plt


def simulate_brownian_motion(T=10, N=100, num_paths=5):
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
    print(t)
    dW = np.random.normal(0, np.sqrt(dt), (num_paths, N)) # mxn - m is num_paths its rows, N is small n columns. 
    print(dW)
    W = np.cumsum(dW, axis=1)
    print("its w\n",W)
    print(W[:, -1] )

    return t, W

# # Generate paths
t, W = simulate_brownian_motion(T=2, N=100, num_paths=5) 

# """
# Graph Interpretation:
# Each path represents a realization of Brownian motion. The spread of paths widens over time, 
# illustrating increasing uncertainty (volatility scales with squareroot of t ).
# """

# Plot
plt.figure(figsize=(30, 5))
for i in range(W.shape[0]):
    plt.plot(t, W[i], label=f'Path {i+1}')
plt.title("Brownian Motion Paths (Filtration Visualization)")
plt.xlabel("Time (t)")
plt.ylabel("W(t)")
plt.legend()
plt.grid(True)
plt.show()
