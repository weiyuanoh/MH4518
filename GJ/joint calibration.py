# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.linalg import cholesky, LinAlgError


# Define the Heston option pricing function
def heston_option_price(kappa, theta, sigma_v, rho, v0, S0, r, q, T, K, option_type='call'):
    """
    Computes the European option price under the Heston model using numerical integration.
    """

    def characteristic_function(phi):
        a = kappa * theta
        u = 0.5
        b = kappa
        sigma = sigma_v

        d = np.sqrt((rho * sigma * 1j * phi - b) ** 2 - sigma ** 2 * (2 * u * 1j * phi - phi ** 2))
        g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)
        exp_dT = np.exp(-d * T)

        C = r * 1j * phi * T + (a / sigma ** 2) * (
                (b - rho * sigma * 1j * phi + d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g))
        )
        D = ((b - rho * sigma * 1j * phi + d) / sigma ** 2) * ((1 - exp_dT) / (1 - g * exp_dT))

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


# Function to read and preprocess price data
def read_and_preprocess_price_data(file_path, asset_name):
    # Load price data
    prices = pd.read_csv(file_path)

    # Convert 'Date' column to datetime
    prices['Date'] = pd.to_datetime(prices['Date'], errors='coerce')
    prices.dropna(subset=['Date'], inplace=True)

    # Identify the price column (adjust as necessary)
    price_columns = ['Close', 'Adj Close', 'Price', 'Last']
    for col in price_columns:
        if col in prices.columns:
            price_col = col
            break
    else:
        raise ValueError(f"No price column found in {asset_name} data.")

    # Rename the price column
    prices.rename(columns={price_col: f'Close_{asset_name}'}, inplace=True)

    # Convert price column to numeric and handle zeros
    prices[f'Close_{asset_name}'] = pd.to_numeric(prices[f'Close_{asset_name}'], errors='coerce')
    prices[f'Close_{asset_name}'].replace(0, np.nan, inplace=True)
    prices.dropna(subset=[f'Close_{asset_name}'], inplace=True)

    return prices[['Date', f'Close_{asset_name}']]


# Read and preprocess price data for Lonza
lonza_price_file = '/Users/guanjuwang/Desktop/TAN /MH4518/lonza.csv'  # Update with your file path
lonza_prices = read_and_preprocess_price_data(lonza_price_file, 'Lonza')

# Read and preprocess price data for Sika
sika_price_file = '/Users/guanjuwang/Desktop/TAN /MH4518/sika.csv'  # Update with your file path
sika_prices = read_and_preprocess_price_data(sika_price_file, 'Sika')

# Merge the data on 'Date'
merged_prices = pd.merge(lonza_prices, sika_prices, on='Date')

# Calculate daily returns
merged_prices['Return_Lonza'] = merged_prices['Close_Lonza'].pct_change()
merged_prices['Return_Sika'] = merged_prices['Close_Sika'].pct_change()

# Drop missing values
merged_prices.dropna(subset=['Return_Lonza', 'Return_Sika'], inplace=True)

# Calculate asset return correlation
rho_S_LONN_S_SIKA = merged_prices['Return_Lonza'].corr(merged_prices['Return_Sika'])
print(f"Correlation between Lonza and Sika returns: {rho_S_LONN_S_SIKA}")

# Calculate variance process correlation using squared returns as proxies
merged_prices['Variance_Lonza'] = merged_prices['Return_Lonza'] ** 2
merged_prices['Variance_Sika'] = merged_prices['Return_Sika'] ** 2

rho_v_LONN_v_SIKA = merged_prices['Variance_Lonza'].corr(merged_prices['Variance_Sika'])
print(f"Correlation between Lonza and Sika variance proxies: {rho_v_LONN_v_SIKA}")

# Import and preprocess market data for Lonza
lonza_option_file = '/Users/guanjuwang/Desktop/TAN /MH4518/lonza.csv'  # Update with your file path
lonza_data = pd.read_csv(lonza_option_file)

# Preprocess Lonza option data
lonza_data.dropna(inplace=True)
lonza_data['Mid'] = (lonza_data['Bid'] + lonza_data['Ask']) / 2
lonza_data['T'] = lonza_data['maturity']  # Assuming 'maturity' is in years
lonza_data['Strike'] = pd.to_numeric(lonza_data['Strike'], errors='coerce')
lonza_data.dropna(subset=['Strike'], inplace=True)

# Ensure 'Date' is in datetime format
lonza_data['Date'] = pd.to_datetime(lonza_data['Date'], errors='coerce')
lonza_data.dropna(subset=['Date'], inplace=True)

# Create market data tuples for Lonza
lonza_market_data = list(zip(lonza_data['Strike'], lonza_data['T'], lonza_data['Mid']))

# Define constants for Lonza
S0_LONN = 549.60  # Replace 'UnderlyingPrice' with correct column
r_LONN = 0.018  # Use average rate if rates vary
q_LONN = 0.0  # Adjust if necessary

# Import and preprocess market data for Sika
sika_option_file = '/Users/guanjuwang/Desktop/TAN /MH4518/sika.csv'  # Update with your file path
sika_data = pd.read_csv(sika_option_file)

# Preprocess Sika option data
sika_data.dropna(inplace=True)
sika_data['Mid'] = (sika_data['Bid'] + sika_data['Ask']) / 2
sika_data['T'] = sika_data['maturity']  # Assuming 'maturity' is in years
sika_data['Strike'] = pd.to_numeric(sika_data['Strike'], errors='coerce')
sika_data.dropna(subset=['Strike'], inplace=True)

# Ensure 'Date' is in datetime format
sika_data['Date'] = pd.to_datetime(sika_data['Date'], errors='coerce')
sika_data.dropna(subset=['Date'], inplace=True)

# Create market data tuples for Sika
sika_market_data = list(zip(sika_data['Strike'], sika_data['T'], sika_data['Mid']))

# Define constants for Sika
S0_SIKA = 240.40  # Replace 'UnderlyingPrice' with correct column
r_SIKA = 0.018  # Use average rate if rates vary
q_SIKA = 0.0  # Adjust if necessary


# Define the joint calibration objective function
def joint_objective_function(params):
    # Unpack parameters
    kappa_LONN, theta_LONN, sigma_v_LONN, rho_S_LONN_v_LONN, v0_LONN, \
        kappa_SIKA, theta_SIKA, sigma_v_SIKA, rho_S_SIKA_v_SIKA, v0_SIKA, \
        rho_S_LONN_S_SIKA, rho_v_LONN_v_SIKA = params

    error_total = 0.0

    # Error for Lonza
    for K, T, market_price in lonza_market_data:
        model_price = heston_option_price(
            kappa_LONN, theta_LONN, sigma_v_LONN, rho_S_LONN_v_LONN, v0_LONN,
            S0_LONN, r_LONN, q_LONN, T, K
        )
        error_total += (model_price - market_price) ** 2

    # Error for Sika
    for K, T, market_price in sika_market_data:
        model_price = heston_option_price(
            kappa_SIKA, theta_SIKA, sigma_v_SIKA, rho_S_SIKA_v_SIKA, v0_SIKA,
            S0_SIKA, r_SIKA, q_SIKA, T, K
        )
        error_total += (model_price - market_price) ** 2

    # Regularization term to ensure covariance matrix is positive definite
    cov_matrix = np.array([
        [1, rho_S_LONN_S_SIKA, rho_S_LONN_v_LONN, 0],
        [rho_S_LONN_S_SIKA, 1, 0, rho_S_SIKA_v_SIKA],
        [rho_S_LONN_v_LONN, 0, 1, rho_v_LONN_v_SIKA],
        [0, rho_S_SIKA_v_SIKA, rho_v_LONN_v_SIKA, 1]
    ])

    try:
        # Attempt Cholesky decomposition to check positive definiteness
        cholesky(cov_matrix)
    except LinAlgError:
        # Penalize the objective function if not positive definite
        error_total += 1e6  # Large penalty
    return error_total


# Initial parameter guesses
initial_params_joint = [
    2.0, 0.02, 0.2, -0.7, 0.02,  # Lonza parameters
    2.0, 0.02, 0.2, -0.7, 0.02,  # Sika parameters
    rho_S_LONN_S_SIKA, rho_v_LONN_v_SIKA  # Correlation parameters
]

# Bounds for all parameters
bounds_joint = [
    # Lonza parameters
    (0.01, 10.0),  # kappa_LONN
    (0.0001, 0.5),  # theta_LONN
    (0.01, 1.0),  # sigma_v_LONN
    (-0.99, 0.0),  # rho_S_LONN_v_LONN
    (0.0001, 0.5),  # v0_LONN
    # Sika parameters
    (0.01, 10.0),  # kappa_SIKA
    (0.0001, 0.5),  # theta_SIKA
    (0.01, 1.0),  # sigma_v_SIKA
    (-0.99, 0.0),  # rho_S_SIKA_v_SIKA
    (0.0001, 0.5),  # v0_SIKA
    # Correlation parameters
    (-0.99, 0.99),  # rho_S_LONN_S_SIKA
    (-0.99, 0.99)  # rho_v_LONN_v_SIKA
]

# Perform the joint calibration
result_joint = minimize(
    joint_objective_function,
    initial_params_joint,
    bounds=bounds_joint,
    method='L-BFGS-B'
)

# Check if optimization was successful
if result_joint.success:
    opt_params = result_joint.x

    # Unpack optimized parameters
    kappa_LONN_opt, theta_LONN_opt, sigma_v_LONN_opt, rho_S_LONN_v_LONN_opt, v0_LONN_opt, \
        kappa_SIKA_opt, theta_SIKA_opt, sigma_v_SIKA_opt, rho_S_SIKA_v_SIKA_opt, v0_SIKA_opt, \
        rho_S_LONN_S_SIKA_opt, rho_v_LONN_v_SIKA_opt = opt_params

    print("\nJoint Calibrated Parameters:")
    print("Lonza Parameters:")
    print(f"kappa: {kappa_LONN_opt}")
    print(f"theta: {theta_LONN_opt}")
    print(f"sigma_v: {sigma_v_LONN_opt}")
    print(f"rho: {rho_S_LONN_v_LONN_opt}")
    print(f"v0: {v0_LONN_opt}")

    print("\nSika Parameters:")
    print(f"kappa: {kappa_SIKA_opt}")
    print(f"theta: {theta_SIKA_opt}")
    print(f"sigma_v: {sigma_v_SIKA_opt}")
    print(f"rho: {rho_S_SIKA_v_SIKA_opt}")
    print(f"v0: {v0_SIKA_opt}")

    print("\nCorrelation Parameters:")
    print(f"rho_S_LONN_S_SIKA: {rho_S_LONN_S_SIKA_opt}")
    print(f"rho_v_LONN_v_SIKA: {rho_v_LONN_v_SIKA_opt}")
else:
    print("Optimization failed.")
    print("Message:", result_joint.message)
