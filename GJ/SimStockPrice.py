import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Dates as da
import constants as cs

#asset parameters
S0 = np.array([549.60, 240.40])  # Initial prices for Lonza and Sika
V0 = np.array([0.010000, 0.195311])      # Initial variances (assumed)
mu = np.array([0.018, 0.018])      # Drift terms (risk-free rate)
kappa = np.array([4.483091, 0.901516])     # Mean reversion speeds
theta = np.array([0.124813, 0.184639])   # Long-term variances
sigma_v = np.array([1, 0.549895])   # Vol of vol
rho = np.array([0.772476, -0.728402])     # Correlation between asset price and its variance


# Correlation between assets' price processes
rho_assets = np.array([[1, 0.4250732997428952],
                       [0.4250732997428952, 1]])

# Simulation parameters
T = 1.25      # Time to maturity (15 months)
N = 314        # Number of time steps (daily steps over 15 months)
nsims = 1000  # Number of simulations


def simulate_heston_multidim(S0, V0, mu, kappa, theta, sigma_v, rho, rho_assets, T, N, nsims):
    """
    Simulate multidimensional Heston model for multiple assets.
    """
    dt = 1/252
    num_assets = len(S0)
    time_grid = np.linspace(0, T, N + 1)

    # Initialize arrays
    S = np.zeros((num_assets, nsims, N + 1))
    V = np.zeros((num_assets, nsims, N + 1))

    # Set initial values
    S[:, :, 0] = S0[:, np.newaxis]
    V[:, :, 0] = V0[:, np.newaxis]

    # Correlation matrices
    corr_price = rho_assets
    corr_vol = np.eye(num_assets)

    # Cholesky decomposition for price and volatilities
    corr_matrix = np.block([
        [corr_price, np.zeros((num_assets, num_assets))],
        [np.zeros((num_assets, num_assets)), corr_vol]
    ])
    L = np.linalg.cholesky(corr_matrix)

    for t in range(1, N + 1):
        # Generate independent standard normal random variables
        Z = np.random.normal(size=(2 * num_assets, nsims))
        # Correlated Brownian increments
        dW = L @ Z * np.sqrt(dt)  # Shape: (2 * num_assets, nsims)

        for i in range(num_assets):
            # Extract correlated increments
            dW_S_i = dW[i, :]
            dW_v_i = dW[num_assets + i, :]
            # Correlate dW_v_i with dW_S_i
            dW_v_i = rho[i] * dW_S_i + np.sqrt(1 - rho[i] ** 2) * dW_v_i

            # Ensure variance remains positive
            V_prev = V[i, :, t - 1]
            V_sqrt = np.sqrt(np.maximum(V_prev, 0))
            V[i, :, t] = V_prev + kappa[i] * (theta[i] - V_prev) * dt + sigma_v[i] * V_sqrt * dW_v_i
            V[i, :, t] = np.maximum(V[i, :, t], 0)

            # Simulate asset price
            S_prev = S[i, :, t - 1]
            S[i, :, t] = S_prev * np.exp((mu[i] - 0.5 * V_prev) * dt + V_sqrt * dW_S_i)

    return S, V, time_grid

S_sim, V_sim, time_grid = simulate_heston_multidim(
    S0, V0, mu, kappa, theta, sigma_v, rho, rho_assets, T, N, nsims)

dates = da.get_list_dates(cs.initial_fixing_date, cs.final_fixing_date)
dates = pd.Series(dates).tolist()

# Convert simulated prices to DataFrames
lonza_paths = pd.DataFrame(S_sim[0].T, index=dates)
sika_paths = pd.DataFrame(S_sim[1].T, index=dates)
