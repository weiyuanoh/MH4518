# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky, LinAlgError
import QuantLib as ql
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ===========================
# 1. Data Preparation
# ===========================

def load_option_data(file_path, asset_name):
    """
    Load and preprocess option data from a CSV file.

    Parameters:
    - file_path: Path to the CSV file.
    - asset_name: Name of the asset (e.g., 'Lonza', 'Sika').

    Returns:
    - df: Pandas DataFrame containing the cleaned option data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    # Read the CSV file with specified date format
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=False, infer_datetime_format=True)
    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {e}")

    # Verify required columns exist
    required_columns = {'maturity', 'Strike', 'Bid', 'Ask', 'Last', 'IVM', 'Volm', 'rate', 'Date'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing columns in {file_path}: {missing}")

    # Add Asset column
    df['Asset'] = asset_name

    # Convert 'Date' to datetime.date objects
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date

    # Drop rows with invalid 'Date'
    df = df.dropna(subset=['Date'])

    # Convert 'maturity' and other columns to numeric, handling errors
    for col in ['maturity', 'Strike', 'IVM', 'rate']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().any():
            print(f"Warning: Missing or invalid values detected in column '{col}' for asset {asset_name}. Dropping these rows.")
            df = df.dropna(subset=[col])

    # Ensure all 'maturity' values are greater than zero
    df = df[df['maturity'] > 0]
    if df.empty:
        print(f"No valid options for asset {asset_name} after removing invalid 'maturity' entries.")
        return None

    # Compute Mid price
    df['Mid'] = (df['Bid'] + df['Ask']) / 2

    # Ensure 'Mid' is numeric and positive
    df['Mid'] = pd.to_numeric(df['Mid'], errors='coerce')
    df = df[df['Mid'] > 0]

    # Final check for required columns
    if df.empty:
        print(f"No valid options for asset {asset_name} after processing.")
        return None

    return df[['Strike', 'maturity', 'Mid']]

def load_price_data(file_path, asset_name):
    """
    Load and preprocess underlying price data from a CSV file.

    Parameters:
    - file_path: Path to the CSV file.
    - asset_name: Name of the asset.

    Returns:
    - df_prices: DataFrame with 'Date' and 'Close_{asset_name}' columns.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    # Read the CSV file with specified date format
    try:
        prices = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=False, infer_datetime_format=True)
    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {e}")

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
    prices[f'Close_{asset_name}'] = prices[f'Close_{asset_name}'].replace(0, np.nan)
    prices = prices.dropna(subset=[f'Close_{asset_name}'])

    return prices[['Date', f'Close_{asset_name}']]

# Paths to your CSV files (update these paths to your actual file locations)
lonza_price_file = '/Users/guanjuwang/Desktop/TAN /MH4518/lonza.csv'  # Update with your actual file path
sika_price_file = '/Users/guanjuwang/Desktop/TAN /MH4518/sika.csv'    # Update with your actual file path
lonza_option_file = '/Users/guanjuwang/Desktop/TAN /MH4518/lonza.csv'  # Update with your actual file path
sika_option_file = '/Users/guanjuwang/Desktop/TAN /MH4518/sika.csv'    # Update with your actual file path

# Load and preprocess price data for Lonza
lonza_prices = load_price_data(lonza_price_file, 'Lonza')

# Load and preprocess price data for Sika
sika_prices = load_price_data(sika_price_file, 'Sika')

# Merge the price data on 'Date'
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

# Read and preprocess market data for Lonza
lonza_option_data = load_option_data(lonza_option_file, 'Lonza')

# Read and preprocess market data for Sika
sika_option_data = load_option_data(sika_option_file, 'Sika')

# Define constants for Lonza and Sika
# Replace these with the actual spot prices and rates if they differ
S0_LONN = 549.60
r_LONN = 0.018  # Risk-free rate for Lonza
q_LONN = 0.0     # Dividend yield for Lonza

S0_SIKA = 240.40
r_SIKA = 0.018    # Risk-free rate for Sika
q_SIKA = 0.0       # Dividend yield for Sika

# ===========================
# 2. Heston Model Calibration
# ===========================

def calibrate_heston(asset_name, option_data, S0, r, q):
    """
    Calibrate the Heston model parameters for a single asset using QuantLib.

    Parameters:
    - asset_name: Name of the asset.
    - option_data: DataFrame with 'Strike', 'maturity', 'Mid' columns.
    - S0: Underlying asset price.
    - r: Risk-free rate.
    - q: Dividend yield.

    Returns:
    - params: Dictionary with calibrated Heston parameters or None.
    """
    print(f"\nStarting calibration for {asset_name}...")

    if option_data is None or option_data.empty:
        print(f"No valid option data available for {asset_name}. Skipping calibration.")
        return None

    # Define evaluation date as today
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today

    # Define day count convention
    day_count = ql.Actual365Fixed()

    # Define flat term structures for risk-free rate and dividends
    risk_free_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, day_count)
    )
    dividend_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(today, q, day_count)
    )

    # Prepare list of options
    options = []
    for index, row in option_data.iterrows():
        T = row['maturity']  # Time to maturity in years
        K = row['Strike']
        market_price = row['Mid']
        option_type = ql.Option.Call  # Assuming all options are calls

        # Calculate maturity date
        maturity_date = today + ql.Period(int(round(T * 365)), ql.Days)

        payoff = ql.PlainVanillaPayoff(option_type, K)
        exercise = ql.EuropeanExercise(maturity_date)
        option = ql.VanillaOption(payoff, exercise)
        # Do not set pricing engine yet
        options.append((option, market_price))

    # Define the calibration objective function
    def objective(params):
        v0, kappa, theta, sigma, rho = params

        # Ensure parameters are within bounds
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or not (-0.99 < rho < 0.99):
            return 1e10  # Large penalty

        # Define Heston process with current parameters
        try:
            heston_process = ql.HestonProcess(
                risk_free_curve,
                dividend_curve,
                ql.QuoteHandle(ql.SimpleQuote(S0)),
                v0,
                kappa,
                theta,
                sigma,
                rho
            )
        except Exception as e:
            print(f"Error initializing HestonProcess: {e}")
            return 1e10

        # Initialize Heston model
        try:
            heston_model = ql.HestonModel(heston_process)
        except Exception as e:
            print(f"Error initializing HestonModel: {e}")
            return 1e10

        # Initialize Heston engine
        try:
            heston_engine = ql.AnalyticHestonEngine(heston_model)
        except Exception as e:
            print(f"Error initializing AnalyticHestonEngine: {e}")
            return 1e10

        # Assign the new engine to each option
        for option, _ in options:
            option.setPricingEngine(heston_engine)

        # Calculate sum of squared errors
        error = 0.0
        for option, market_price in options:
            try:
                model_price = option.NPV()
                if not np.isfinite(model_price) or model_price <= 0:
                    print(f"Invalid model price {model_price} for {asset_name}, Strike {option.payoff().strike()}, Maturity {option.exercise().lastDate()}")
                    return 1e10
                error += (model_price - market_price) ** 2
            except Exception as e:
                print(f"Error pricing option for {asset_name}: {e}")
                return 1e10

        return error

    # Initial parameter guesses: [v0, kappa, theta, sigma, rho]
    initial_guess = [0.04, 1.0, 0.04, 0.5, -0.5]

    # Define bounds for parameters
    bounds = [
        (1e-4, 5.0),    # v0
        (1e-4, 10.0),   # kappa
        (1e-4, 1.0),    # theta
        (1e-4, 1.0),    # sigma
        (-0.99, 0.99)   # rho
    ]

    # Perform calibration using SciPy's minimize function
    result = minimize(
        objective,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': True, 'maxiter': 1000}
    )

    if result.success:
        calibrated_params = {
            'v0': result.x[0],
            'kappa': result.x[1],
            'theta': result.x[2],
            'sigma': result.x[3],
            'rho': result.x[4]
        }
        print(f"\nCalibration successful for {asset_name}: {calibrated_params}")
        return calibrated_params
    else:
        print(f"\nCalibration failed for {asset_name}: {result.message}")
        return None

# ===========================
# 3. Joint Calibration Setup
# ===========================

# Define joint calibration objective function
def joint_objective_function(params, lonza_market_data, sika_market_data, rho_S_LONN_S_SIKA, rho_v_LONN_v_SIKA):
    """
    Objective function for joint calibration of Lonza and Sika Heston models with covariance constraints.

    Parameters:
    - params: List of parameters.
    - lonza_market_data: List of tuples (option, market_price) for Lonza.
    - sika_market_data: List of tuples (option, market_price) for Sika.
    - rho_S_LONN_S_SIKA: Correlation between Lonza and Sika returns.
    - rho_v_LONN_v_SIKA: Correlation between Lonza and Sika variance proxies.

    Returns:
    - error_total: Sum of squared errors plus penalty for covariance matrix.
    """
    # Unpack parameters
    kappa_LONN, theta_LONN, sigma_v_LONN, rho_S_LONN_v_LONN, v0_LONN, \
    kappa_SIKA, theta_SIKA, sigma_v_SIKA, rho_S_SIKA_v_SIKA, v0_SIKA = params

    # Initialize Heston models for Lonza and Sika
    try:
        # Lonza HestonProcess
        heston_process_lonza = ql.HestonProcess(
            risk_free_curve_lonza,
            dividend_curve_lonza,
            ql.QuoteHandle(ql.SimpleQuote(S0_LONN)),
            v0_LONN,
            kappa_LONN,
            theta_LONN,
            sigma_v_LONN,
            rho_S_LONN_v_LONN
        )
        heston_model_lonza = ql.HestonModel(heston_process_lonza)
        heston_engine_lonza = ql.AnalyticHestonEngine(heston_model_lonza)
    except Exception as e:
        print(f"Error initializing Lonza HestonModel: {e}")
        return 1e10

    try:
        # Sika HestonProcess
        heston_process_sika = ql.HestonProcess(
            risk_free_curve_sika,
            dividend_curve_sika,
            ql.QuoteHandle(ql.SimpleQuote(S0_SIKA)),
            v0_SIKA,
            kappa_SIKA,
            theta_SIKA,
            sigma_v_SIKA,
            rho_S_SIKA_v_SIKA
        )
        heston_model_sika = ql.HestonModel(heston_process_sika)
        heston_engine_sika = ql.AnalyticHestonEngine(heston_model_sika)
    except Exception as e:
        print(f"Error initializing Sika HestonModel: {e}")
        return 1e10

    # Assign engines to options
    for option, _ in lonza_market_data:
        option.setPricingEngine(heston_engine_lonza)
    for option, _ in sika_market_data:
        option.setPricingEngine(heston_engine_sika)

    # Calculate sum of squared errors
    error_total = 0.0
    for option, market_price in lonza_market_data:
        try:
            model_price = option.NPV()
            if not np.isfinite(model_price) or model_price <= 0:
                print(f"Invalid model price {model_price} for Lonza, Strike {option.payoff().strike()}, Maturity {option.exercise().lastDate()}")
                return 1e10
            error_total += (model_price - market_price) ** 2
        except Exception as e:
            print(f"Error pricing Lonza option: {e}")
            return 1e10

    for option, market_price in sika_market_data:
        try:
            model_price = option.NPV()
            if not np.isfinite(model_price) or model_price <= 0:
                print(f"Invalid model price {model_price} for Sika, Strike {option.payoff().strike()}, Maturity {option.exercise().lastDate()}")
                return 1e10
            error_total += (model_price - market_price) ** 2
        except Exception as e:
            print(f"Error pricing Sika option: {e}")
            return 1e10

    # Regularization: Ensure covariance matrix is positive definite
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

# ===========================
# 3. Perform Calibration
# ===========================

# Calibrate Heston model for Lonza
def calibrate_individual(asset_name, option_data, S0, r, q):
    """
    Calibrate the Heston model for an individual asset.

    Parameters:
    - asset_name: Name of the asset.
    - option_data: DataFrame with 'Strike', 'maturity', 'Mid' columns.
    - S0: Underlying asset price.
    - r: Risk-free rate.
    - q: Dividend yield.

    Returns:
    - calibrated_params: Dictionary with calibrated Heston parameters or None.
    """
    return calibrate_heston(asset_name, option_data, S0, r, q)

# Calibrate Lonza
calibrated_params_lonza = calibrate_individual(
    asset_name='Lonza',
    option_data=lonza_option_data,
    S0=S0_LONN,
    r=r_LONN,
    q=q_LONN
)

# Calibrate Sika
calibrated_params_sika = calibrate_individual(
    asset_name='Sika',
    option_data=sika_option_data,
    S0=S0_SIKA,
    r=r_SIKA,
    q=q_SIKA
)

# ===========================
# 4. Display Calibrated Parameters
# ===========================

print("\nCalibrated Heston Model Parameters:")
if calibrated_params_lonza:
    print("Lonza:")
    for param, value in calibrated_params_lonza.items():
        print(f"  {param}: {value:.6f}")
    print()
if calibrated_params_sika:
    print("Sika:")
    for param, value in calibrated_params_sika.items():
        print(f"  {param}: {value:.6f}")
    print()
