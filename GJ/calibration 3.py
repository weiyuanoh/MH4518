import pandas as pd  # Fixed typo from 't pandas as pd'
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky, LinAlgError
from scipy.integrate import quad  # Ensure integration is imported
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
lonza_price_file = '/Users/guanjuwang/Desktop/TAN /MH4518/lonza_price.csv'  # Update with your actual file path
sika_price_file = '/Users/guanjuwang/Desktop/TAN /MH4518/sika_price.csv'    # Update with your actual file path
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

S0_SIKA = 240.40
r_SIKA = 0.018    # Risk-free rate for Sika

# ===========================
# 2. Heston Model Calibration
# ===========================

def heston_option_price(kappa, theta, sigma_v, rho, v0, S0, r, T, K, option_type='call'):
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

        return np.exp(C + D * v0 + 1j * phi * np.log(S0))

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
        price = S0 * P1 - K * np.exp(-r * T) * P2
    elif option_type.lower() == 'put':
        price = K * np.exp(-r * T) * (1 - P2) - S0 * (1 - P1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price

def calibrate_heston(asset_name, option_data, S0, r):
    """
    Calibrate the Heston model parameters for a single asset using QuantLib.

    Parameters:
    - asset_name: Name of the asset.
    - option_data: DataFrame with 'Strike', 'maturity', 'Mid' columns.
    - S0: Underlying asset price.
    - r: Risk-free rate.

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

    # Define flat term structures for risk-free rate (no dividends)
    risk_free_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, day_count)
    )
    dividend_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(today, 0.0, day_count)  # Set dividend yield to 0
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
        options.append((option, market_price, K))

    # Define the calibration objective function
    def calibration_objective(params):
        v0, kappa, theta, sigma, rho = params

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
            return np.inf  # Discourage optimizer from this path

        # Initialize Heston model
        try:
            heston_model = ql.HestonModel(heston_process)
        except Exception as e:
            print(f"Error initializing HestonModel: {e}")
            return np.inf  # Discourage optimizer from this path

        # Initialize Heston engine
        try:
            heston_engine = ql.AnalyticHestonEngine(heston_model)
        except Exception as e:
            print(f"Error initializing AnalyticHestonEngine: {e}")
            return np.inf  # Discourage optimizer from this path

        # Assign the new engine to each option
        for option, market_price, K in options:
            option.setPricingEngine(heston_engine)

        # Calculate sum of squared errors
        error = 0.0
        for option, market_price, K in options:
            try:
                model_price = option.NPV()
                error += (model_price - market_price) ** 2  # No penalties, include as-is
            except Exception as e:
                print(f"Error pricing option for {asset_name}: {e}")
                return np.inf  # Discourage optimizer from this path

        return error

    # Initial parameter guesses: [v0, kappa, theta, sigma, rho]
    initial_guess = [0.2, 1.0, 0.1, 0.5, -0.7]  # Adjusted initial guess for theta

    # Define bounds for parameters
    bounds = [
        (0.01, 1.0),  # v0
        (0.01, 5.0),  # kappa
        (0.01, 1.0),  # theta
        (0.01, 1.0),  # sigma
        (-0.99, 0.99)  # rho
    ]

    # Perform calibration using SciPy's minimize function
    result = minimize(
        calibration_objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        options={'ftol': 1e-6, 'maxiter': 500}
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

# Since both assets do not pay dividends, no need to handle dividend-related variables.
# The joint calibration function can be simplified accordingly.

def joint_objective_function(params, lonza_market_data, sika_market_data, rho_S_LONN_S_SIKA, rho_v_LONN_v_SIKA):
    """
    Objective function for joint calibration of Lonza and Sika Heston models.

    Parameters:
    - params: List of parameters.
    - lonza_market_data: List of tuples (option, market_price, K) for Lonza.
    - sika_market_data: List of tuples (option, market_price, K) for Sika.
    - rho_S_LONN_S_SIKA: Correlation between Lonza and Sika returns.
    - rho_v_LONN_v_SIKA: Correlation between Lonza and Sika variance proxies.

    Returns:
    - error_total: Sum of squared errors.
    """
    # Unpack parameters
    kappa_LONN, theta_LONN, sigma_v_LONN, rho_S_LONN_v_LONN, v0_LONN, \
    kappa_SIKA, theta_SIKA, sigma_v_SIKA, rho_S_SIKA_v_SIKA, v0_SIKA = params

    # Initialize Heston models for Lonza and Sika
    try:
        # Lonza HestonProcess
        heston_process_lonza = ql.HestonProcess(
            risk_free_curve_lonza,  # These need to be defined in the broader scope
            dividend_curve_lonza,    # Set to zero as dividends are not paid
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
        return np.inf  # Removed large penalty

    try:
        # Sika HestonProcess
        heston_process_sika = ql.HestonProcess(
            risk_free_curve_sika,  # These need to be defined in the broader scope
            dividend_curve_sika,    # Set to zero as dividends are not paid
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
        return np.inf  # Removed large penalty

    # Assign engines to options
    for option, market_price, K in lonza_market_data:
        option.setPricingEngine(heston_engine_lonza)
    for option, market_price, K in sika_market_data:
        option.setPricingEngine(heston_engine_sika)

    # Calculate sum of squared errors
    error_total = 0.0
    for option, market_price, K in lonza_market_data:
        try:
            model_price = option.NPV()
            # No penalty for negative model prices; include them as is
            error_total += (model_price - market_price) ** 2
        except Exception as e:
            print(f"Error pricing Lonza option: {e}")
            return np.inf  # Removed large penalty

    for option, market_price, K in sika_market_data:
        try:
            model_price = option.NPV()
            # No penalty for negative model prices; include them as is
            error_total += (model_price - market_price) ** 2
        except Exception as e:
            print(f"Error pricing Sika option: {e}")
            return np.inf  # Removed large penalty

    # Regularization: Removed as covariance constraints are no longer imposed

    return error_total

# ===========================
# 4. Perform Calibration
# ===========================

# Calibrate Heston model for Lonza
def calibrate_individual(asset_name, option_data, S0, r):
    """
    Calibrate the Heston model for an individual asset.

    Parameters:
    - asset_name: Name of the asset.
    - option_data: DataFrame with 'Strike', 'maturity', 'Mid' columns.
    - S0: Underlying asset price.
    - r: Risk-free rate.

    Returns:
    - calibrated_params: Dictionary with calibrated Heston parameters or None.
    """
    return calibrate_heston(asset_name, option_data, S0, r)

# Calibrate Lonza
calibrated_params_lonza = calibrate_individual(
    asset_name='Lonza',
    option_data=lonza_option_data,
    S0=S0_LONN,
    r=r_LONN
)

# Calibrate Sika
calibrated_params_sika = calibrate_individual(
    asset_name='Sika',
    option_data=sika_option_data,
    S0=S0_SIKA,
    r=r_SIKA
)

# ===========================
# 5. Display Calibrated Parameters
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

