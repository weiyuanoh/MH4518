import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# Simulation parameters
S0_LONN = 549.60       # Initial price of Lonza Group AG
S0_SIKA = 240.40       # Initial price of Sika AG
T = 1.25               # Time to maturity in years
r = 0.018               # Risk-free rate
N = 315                # Number of time steps in simulation
M = 10000              # Number of simulation paths

# Calibrated Heston parameters for Lonza Group AG
kappa_LONN = 3.0
theta_LONN = 0.20**2
v0_LONN = 0.25**2
rho_S_LONN_v_LONN = -0.7
sigma_v_LONN = 0.6

# Calibrated Heston parameters for Sika AG
kappa_SIKA = 2.5
theta_SIKA = 0.25**2
v0_SIKA = 0.30**2
rho_S_SIKA_v_SIKA = -0.6
sigma_v_SIKA = 0.5

# Correlations between assets
rho_S_LONN_S_SIKA = 0.5      # Correlation between asset prices
rho_v_LONN_v_SIKA = 0.3      # Correlation between variances

def simulate_two_asset_heston(S0_LONN, S0_SIKA, v0_LONN, v0_SIKA, params, T, N, M):
    dt = T / N
    mu = np.zeros(4)

        # Construct the covariance matrix
    cov_matrix = np.array([
        [1, params['rho_S_LONN_S_SIKA'], params['rho_S_LONN_v_LONN'], 0],
        [params['rho_S_LONN_S_SIKA'], 1, 0, params['rho_S_SIKA_v_SIKA']],
        [params['rho_S_LONN_v_LONN'], 0, 1, params['rho_v_LONN_v_SIKA']],
        [0, params['rho_S_SIKA_v_SIKA'], params['rho_v_LONN_v_SIKA'], 1]
    ])

    # Check if the matrix is positive definite
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # If not, make it positive definite by adding a small diagonal perturbation
        print("Warning: Covariance matrix is not positive definite. Applying a small perturbation.")
        min_eig = np.min(np.linalg.eigvals(cov_matrix))
        perturbation = np.abs(min_eig) + 1e-6  # Add a small value to the minimum eigenvalue
        cov_matrix += perturbation * np.eye(cov_matrix.shape[0])
        L = np.linalg.cholesky(cov_matrix)  # Retry Cholesky decomposition

    # Initialize arrays
    S_LONN = np.full((N+1, M), S0_LONN)
    v_LONN = np.full((N+1, M), v0_LONN)
    S_SIKA = np.full((N+1, M), S0_SIKA)
    v_SIKA = np.full((N+1, M), v0_SIKA)

    for i in range(1, N+1):
        # Generate correlated Brownian increments
        Z = np.random.normal(size=(M, 4))
        dW = Z @ L.T * np.sqrt(dt)

        # Lonza Group AG
        v_LONN_prev = v_LONN[i-1]
        v_LONN_sqrt = np.sqrt(np.maximum(v_LONN_prev, 0))
        v_LONN[i] = v_LONN_prev + params['kappa_LONN'] * (params['theta_LONN'] - v_LONN_prev) * dt + params['sigma_v_LONN'] * v_LONN_sqrt * dW[:, 2]
        v_LONN[i] = np.maximum(v_LONN[i], 0)
        S_LONN[i] = S_LONN[i-1] * np.exp((r - 0.5 * v_LONN_prev) * dt + v_LONN_sqrt * dW[:, 0])

        # Sika AG
        v_SIKA_prev = v_SIKA[i-1]
        v_SIKA_sqrt = np.sqrt(np.maximum(v_SIKA_prev, 0))
        v_SIKA[i] = v_SIKA_prev + params['kappa_SIKA'] * (params['theta_SIKA'] - v_SIKA_prev) * dt + params['sigma_v_SIKA'] * v_SIKA_sqrt * dW[:, 3]
        v_SIKA[i] = np.maximum(v_SIKA[i], 0)
        S_SIKA[i] = S_SIKA[i-1] * np.exp((r - 0.5 * v_SIKA_prev) * dt + v_SIKA_sqrt * dW[:, 1])

    return S_LONN, v_LONN, S_SIKA, v_SIKA

# Parameters dictionary
params = {
    'kappa_LONN': kappa_LONN,
    'theta_LONN': theta_LONN,
    'sigma_v_LONN': sigma_v_LONN,
    'rho_S_LONN_v_LONN': rho_S_LONN_v_LONN,
    'kappa_SIKA': kappa_SIKA,
    'theta_SIKA': theta_SIKA,
    'sigma_v_SIKA': sigma_v_SIKA,
    'rho_S_SIKA_v_SIKA': rho_S_SIKA_v_SIKA,
    'rho_S_LONN_S_SIKA': rho_S_LONN_S_SIKA,
    'rho_v_LONN_v_SIKA': rho_v_LONN_v_SIKA
}

# Simulate the two assets
print("Starting asset path simulation...")
S_LONN, v_LONN, S_SIKA, v_SIKA = simulate_two_asset_heston(S0_LONN, S0_SIKA, v0_LONN, v0_SIKA, params, T, N, M)
print("Asset path simulation completed.")

# Product parameters
params_product = {
    'Denomination': 1000,
    'Coupon_Rate': 0.08 / 4,  # Quarterly coupon payments
    'Barrier_LONN': 0.6 * S0_LONN,  # 60% barrier for Lonza
    'Barrier_SIKA': 0.6 * S0_SIKA,  # 60% barrier for Sika
    'Early_Redemption_Level_LONN': S0_LONN,
    'Early_Redemption_Level_SIKA': S0_SIKA,
    'S0_LONN': S0_LONN,
    'S0_SIKA': S0_SIKA,
    'T': T,
    'early_observation_times': [0.25, 0.5, 0.75, 1.0, 1.25],
}

def compute_payoff_paths(S_LONN_paths, S_SIKA_paths, params):
    M = S_LONN_paths.shape[1]
    N = S_LONN_paths.shape[0] - 1
    dt = params['T'] / N
    time_grid = np.linspace(0, params['T'], N+1)
    early_indices = [np.searchsorted(time_grid, t) for t in params['early_observation_times']]
    cash_flows = np.zeros((N+1, M))  # Cash flows at each time step

    for i in range(M):
        early_redeemed = False
        S_LONN = S_LONN_paths[:, i]
        S_SIKA = S_SIKA_paths[:, i]

        for idx, t_idx in enumerate(early_indices):
            if (S_LONN[t_idx] >= params['Early_Redemption_Level_LONN'] and
                S_SIKA[t_idx] >= params['Early_Redemption_Level_SIKA']):
                periods = idx + 1
                cash_flow = params['Denomination'] * (1 + params['Coupon_Rate'] * periods)
                cash_flows[t_idx, i] = cash_flow
                # Set future cash flows to zero since the product is redeemed
                cash_flows[t_idx+1:, i] = 0
                early_redeemed = True
                break

        if not early_redeemed:
            barrier_breached = np.any((S_LONN <= params['Barrier_LONN']) | (S_SIKA <= params['Barrier_SIKA']))
            final_S_LONN = S_LONN[-1]
            final_S_SIKA = S_SIKA[-1]
            periods = len(params['early_observation_times'])
            t_idx = N  # Maturity index

            if not barrier_breached and (final_S_LONN >= params['S0_LONN'] and final_S_SIKA >= params['S0_SIKA']):
                cash_flow = params['Denomination'] * (1 + params['Coupon_Rate'] * periods)
            else:
                performance_LONN = final_S_LONN / params['S0_LONN']
                performance_SIKA = final_S_SIKA / params['S0_SIKA']
                worst_performance = min(performance_LONN, performance_SIKA)
                cash_flow = params['Denomination'] * worst_performance + params['Denomination'] * params['Coupon_Rate'] * periods
            cash_flows[t_idx, i] = cash_flow

    return cash_flows

def compute_product_price_paths(S_LONN_paths, S_SIKA_paths, params):
    M = S_LONN_paths.shape[1]
    N = S_LONN_paths.shape[0] - 1
    dt = params['T'] / N
    time_grid = np.linspace(0, params['T'], N+1)
    early_indices = [np.searchsorted(time_grid, t) for t in params['early_observation_times']]
    cash_flows = compute_payoff_paths(S_LONN_paths, S_SIKA_paths, params)

    # Initialize the continuation value matrix
    V = np.zeros((N+1, M))

    # Set the terminal values
    V[-1] = cash_flows[-1]

    # Backward induction
    for t in reversed(range(N)):
        # Discount future value to time t
        V[t] = np.exp(-r * dt) * V[t+1]

        # If t is an early observation date
        if t in early_indices:
            # Cash flow at time t
            cf = cash_flows[t]
            # Decide whether to redeem or continue
            exercise = cf > V[t]
            # Update value
            V[t] = np.where(exercise, cf, V[t])

    return V

# Compute the product's price paths
print("Starting product price computation...")
V = compute_product_price_paths(S_LONN, S_SIKA, params_product)
print("Product price computation completed.")

# Extract the product's price at time 0 for each path
product_prices = V[0]

# Calculate the expected price
expected_price = np.mean(product_prices)
print(f"The expected price of the product is: {expected_price:.2f}")

# Analyze the distribution
plt.hist(product_prices, bins=50)
plt.title('Distribution of Simulated Product Prices at Time 0')
plt.xlabel('Product Price')
plt.ylabel('Frequency')
plt.show()

# Analyze the average product price over time
time_grid = np.linspace(0, T, N+1)
average_prices_over_time = np.mean(V, axis=1)
plt.plot(time_grid, average_prices_over_time)
plt.title('Average Product Price Over Time')
plt.xlabel('Time (Years)')
plt.ylabel('Product Price')
plt.grid(True)
plt.show()

# Plotting the asset prices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
fig, ax3 = plt.subplots(figsize=(12, 6))
time = np.linspace(0, T, N+1)

# Plot sample paths (only plotting the first 100 paths to reduce clutter)
ax1.plot(time, S_LONN[:, :100], color='blue', alpha=0.1)
ax1.set_title('LONZA Prices under Multidimensional Heston Model')
ax1.set_xlabel('Time')
ax1.set_ylabel('LONZA Price')

ax2.plot(time, S_SIKA[:, :100], color='green', alpha=0.1)
ax2.set_title('SIKA Prices under Multidimensional Heston Model')
ax2.set_xlabel('Time')
ax2.set_ylabel('SIKA Price')

ax3.plot(time, V[:, :100], color='red', alpha =0.1)
ax3.set_title('Simulated Paths Heston Model')
ax3.set_xlabel('Time')
ax3.set_ylabel('Simulated Paths')

plt.tight_layout()
plt.show()
