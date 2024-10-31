# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

# Define the Heston option pricing function
def heston_option_price(kappa, theta, sigma_v, rho, v0, S0, r, q, T, K, option_type='call'):
    """
    Computes the European option price under the Heston model using numerical integration.

    Parameters:
        kappa (float): Mean reversion rate of the variance process.
        theta (float): Long-term variance (mean of the variance process).
        sigma_v (float): Volatility of the variance process ('vol of vol').
        rho (float): Correlation between the Brownian motions of the asset and its variance.
        v0 (float): Initial variance.
        S0 (float): Current asset price.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        T (float): Time to maturity (in years).
        K (float): Strike price.
        option_type (str): 'call' or 'put'.

    Returns:
        price (float): Price of the European option under the Heston model.
    """
    def characteristic_function(phi):
        a = kappa * theta
        u = 0.5
        b = kappa
        sigma = sigma_v

        d = np.sqrt((rho * sigma * 1j * phi - b)**2 - sigma**2 * (2 * u * 1j * phi - phi**2))
        g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)
        exp_dT = np.exp(-d * T)

        C = r * 1j * phi * T + (a / sigma**2) * ((b - rho * sigma * 1j * phi + d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g)))
        D = ((b - rho * sigma * 1j * phi + d) / sigma**2) * ((1 - exp_dT) / (1 - g * exp_dT))

        return np.exp(C + D * v0 + 1j * phi * np.log(S0 * np.exp(-q * T)))

    def integrand(phi, Pnum):
        cf = characteristic_function(phi - (Pnum - 1) * 1j)
        numerator = np.exp(-1j * phi * np.log(K)) * cf
        denominator = 1j * phi
        return (numerator / denominator).real

    limit = 150  # Upper limit for integration

    # Compute probabilities P1 and P2
    P1 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, Pnum=1), 0, limit, limit=1000)[0]
    P2 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, Pnum=2), 0, limit, limit=1000)[0]

    # Option price calculation
    if option_type.lower() == 'call':
        price = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    elif option_type.lower() == 'put':
        price = K * np.exp(-r * T) * (1 - P2) - S0 * np.exp(-q * T) * (1 - P1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price

# Import and preprocess market data for Lonza
lonza_file_path = '/Users/guanjuwang/Desktop/TAN /MH4518/lonza.csv'  # Update with your file path
lonza_data = pd.read_csv(lonza_file_path)

# Preprocess Lonza data
lonza_data.dropna(inplace=True)
lonza_data['Mid'] = (lonza_data['Bid'] + lonza_data['Ask']) / 2
lonza_data['T'] = lonza_data['maturity']  # Assuming 'maturity' is in years
lonza_data['Strike'] = pd.to_numeric(lonza_data['Strike'])

# Ensure 'Date' is in datetime format
lonza_data['Date'] = pd.to_datetime(lonza_data['Date'], errors='coerce')
lonza_data.dropna(subset=['Date'], inplace=True)

# Create market data tuples for Lonza
lonza_market_data = list(zip(lonza_data['Strike'], lonza_data['T'], lonza_data['Mid']))

# Define constants for Lonza
S0_LONN = 549.60  # Replace 'UnderlyingPrice' with correct column
r_LONN = 0.018  # Use average rate if rates vary
q_LONN = 0.0  # Adjust if necessary

# Define the objective function for Lonza
def lonza_objective_function(params):
    kappa, theta, sigma_v, rho, v0 = params
    error = 0.0
    for K, T, market_price in lonza_market_data:
        model_price = heston_option_price(kappa, theta, sigma_v, rho, v0, S0_LONN, r_LONN, q_LONN, T, K)
        error += (model_price - market_price) ** 2
    return error

# Initial parameter guess for Lonza
initial_params_lonza = [2.0, 0.02, 0.2, -0.7, 0.02]

# Bounds for Lonza's parameters
bounds_lonza = [
    (0.01, 10.0),   # kappa
    (0.0001, 0.5),  # theta
    (0.01, 1.0),    # sigma_v
    (-0.99, 0.0),   # rho
    (0.0001, 0.5)   # v0
]

# Perform calibration for Lonza
result_lonza = minimize(lonza_objective_function, initial_params_lonza, bounds=bounds_lonza, method='L-BFGS-B')

# Extract calibrated parameters for Lonza
kappa_LONN_opt, theta_LONN_opt, sigma_v_LONN_opt, rho_S_LONN_v_LONN_opt, v0_LONN_opt = result_lonza.x

print("Calibrated Parameters for Lonza:")
print(f"kappa: {kappa_LONN_opt}")
print(f"theta: {theta_LONN_opt}")
print(f"sigma_v: {sigma_v_LONN_opt}")
print(f"rho: {rho_S_LONN_v_LONN_opt}")
print(f"v0: {v0_LONN_opt}")

# Import and preprocess market data for Sika
sika_file_path = '/Users/guanjuwang/Desktop/TAN /MH4518/sika.csv'  # Update with your file path
sika_data = pd.read_csv(sika_file_path)

# Preprocess Sika data
sika_data.dropna(inplace=True)
sika_data['Mid'] = (sika_data['Bid'] + sika_data['Ask']) / 2
sika_data['T'] = sika_data['maturity']  # Assuming 'maturity' is in years
sika_data['Strike'] = pd.to_numeric(sika_data['Strike'])

# Ensure 'Date' is in datetime format
sika_data['Date'] = pd.to_datetime(sika_data['Date'], errors='coerce')
sika_data.dropna(subset=['Date'], inplace=True)

# Create market data tuples for Sika
sika_market_data = list(zip(sika_data['Strike'], sika_data['T'], sika_data['Mid']))

# Define constants for Sika
S0_SIKA = 240.4  # Replace 'UnderlyingPrice' with correct column
r_SIKA = 0.018  # Use average rate if rates vary
q_SIKA = 0.0  # Adjust if necessary

# Define the objective function for Sika
def sika_objective_function(params):
    kappa, theta, sigma_v, rho, v0 = params
    error = 0.0
    for K, T, market_price in sika_market_data:
        model_price = heston_option_price(kappa, theta, sigma_v, rho, v0, S0_SIKA, r_SIKA, q_SIKA, T, K)
        error += (model_price - market_price) ** 2
    return error

# Initial parameter guess for Sika
initial_params_sika = [2.0, 0.02, 0.2, -0.7, 0.02]

# Bounds for Sika's parameters
bounds_sika = [
    (0.01, 10.0),   # kappa
    (0.0001, 0.5),  # theta
    (0.01, 1.0),    # sigma_v
    (-0.99, 0.0),   # rho
    (0.0001, 0.5)   # v0
]

# Perform calibration for Sika
result_sika = minimize(sika_objective_function, initial_params_sika, bounds=bounds_sika, method='L-BFGS-B')

# Extract calibrated parameters for Sika
kappa_SIKA_opt, theta_SIKA_opt, sigma_v_SIKA_opt, rho_S_SIKA_v_SIKA_opt, v0_SIKA_opt = result_sika.x

print("\nCalibrated Parameters for Sika:")
print(f"kappa: {kappa_SIKA_opt}")
print(f"theta: {theta_SIKA_opt}")
print(f"sigma_v: {sigma_v_SIKA_opt}")
print(f"rho: {rho_S_SIKA_v_SIKA_opt}")
print(f"v0: {v0_SIKA_opt}")


