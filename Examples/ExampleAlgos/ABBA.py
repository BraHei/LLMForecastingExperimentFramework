import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# Simulation parameters
T = 1.0   # Time horizon
N = 100   # Number of time steps
dt = T / N  # Time step size
x0 = 0.0  # Starting value
xT = 1.0  # Ending value
adaptive_factor = 0.5  # Adaptive control factor for variance adjustment

# # Time grid
# t = np.linspace(0, T, N+1)

# # Standard Brownian motion
# W = np.random.randn(N) * np.sqrt(dt)
# W = np.insert(np.cumsum(W), 0, 0)  # Insert initial value

# # Adaptive Brownian Bridge
# X = np.zeros(N+1)
# X[0] = x0

# for i in range(1, N+1):
#     adaptive_variance = adaptive_factor * (T - t[i])  # Adaptive variance scaling
#     drift = (xT - X[i-1]) / (T - t[i-1]) if t[i-1] != T else 0  # Adaptive drift
#     X[i] = X[i-1] + drift * dt + np.sqrt(adaptive_variance * dt) * np.random.randn()

# # Plot the Adaptive Brownian Bridge
# plt.figure(figsize=(10, 5))
# plt.plot(t, X, label="Adaptive Brownian Bridge", linewidth=2)
# plt.axhline(y=xT, color='r', linestyle='--', label="Target $x_T$")
# plt.axhline(y=x0, color='g', linestyle='--', label="Start $x_0$")
# plt.xlabel("Time")
# plt.ylabel("Process Value")
# plt.title("Adaptive Brownian Bridge Simulation")
# plt.legend()
# plt.grid()
# plt.show()

# # Re-run the simulation and display the Adaptive Brownian Bridge with a more refined visualization

# # Generate new Brownian motion
# W = np.random.randn(N) * np.sqrt(dt)
# W = np.insert(np.cumsum(W), 0, 0)  # Insert initial value

# # Adaptive Brownian Bridge Simulation
# X = np.zeros(N+1)
# X[0] = x0

# for i in range(1, N+1):
#     adaptive_variance = adaptive_factor * (T - t[i])  # Adaptive variance scaling
#     drift = (xT - X[i-1]) / (T - t[i-1]) if t[i-1] != T else 0  # Adaptive drift
#     X[i] = X[i-1] + drift * dt + np.sqrt(adaptive_variance * dt) * np.random.randn()

# # Improved visualization
# plt.figure(figsize=(10, 5))
# plt.plot(t, X, label="Adaptive Brownian Bridge", linewidth=2, color='blue')
# plt.scatter([0, T], [x0, xT], color='red', zorder=3, label="Endpoints ($x_0, x_T$)")
# plt.fill_between(t, X - 0.1, X + 0.1, color='blue', alpha=0.2, label="Variance Range")
# plt.axhline(y=xT, color='r', linestyle='--', label="Target $x_T$")
# plt.axhline(y=x0, color='g', linestyle='--', label="Start $x_0$")
# plt.xlabel("Time")
# plt.ylabel("Process Value")
# plt.title("Adaptive Brownian Bridge Simulation")
# plt.legend()
# plt.grid()
# plt.show()

# Generate a synthetic time series (e.g., a noisy sinusoidal pattern)
T_series = np.linspace(0, 10, N+1)
Y_series = np.sin(T_series) + 0.2 * np.random.randn(N+1)  # Add noise

# Adaptive Brownian Bridge segmentation
num_segments = 10  # Define number of symbolic segments
segment_points = np.linspace(0, N, num_segments+1, dtype=int)  # Define segment indices

print(segment_points)

# Compute Brownian Bridge approximations for each segment
X_bridge = np.zeros(N+1)
for i in range(len(segment_points) - 1):
    start, end = segment_points[i], segment_points[i+1]
    X_bridge[start:end+1] = np.linspace(Y_series[start], Y_series[end], end-start+1)  # Linear bridge

# Apply a symbolic approximation using ABBA-like segmentation
symbolic_segments = scipy.signal.find_peaks(np.abs(Y_series - X_bridge), distance=N//num_segments)[0]

# Plot original time series vs. Brownian Bridge Approximation
plt.figure(figsize=(12, 6))
plt.plot(T_series, Y_series, label="Original Time Series", linewidth=1.5, color='gray', alpha=0.7)
plt.plot(T_series, X_bridge, label="Brownian Bridge Approximation", linewidth=2, color='blue')
plt.scatter(T_series[symbolic_segments], X_bridge[symbolic_segments], color='red', marker='o', label="Symbolic Breakpoints", zorder=3)
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Adaptive Brownian Bridge for Symbolic Time Series Approximation")
plt.legend()
plt.grid()
plt.show()