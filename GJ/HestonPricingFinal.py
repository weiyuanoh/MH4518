import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Define constants
class Constants:
    initial_fixing_date = "2023-04-27"
    final_fixing_date = "2024-07-30"
    initialS1 = 549.60  # Lonza initial price
    initialS2 = 240.40  # Sika initial price
    barrierS1 = 329.76
    barrierS2 = 144.24
    conversionratioS1 = 1.8195
    conversionratioS2 = 4.1597
    initial_investment = 1000  # CHF


# Define Dates functions
class Dates:
    @staticmethod
    def get_list_dates(start_date, end_date):
        """
        Generate a list of business dates between start_date and end_date.
        """
        return pd.date_range(start=start_date, end=end_date, freq='B')  # Business days

    @staticmethod
    def get_business_days(start_date, end_date):
        """
        Generate a list of business dates between start_date and end_date.
        """
        return pd.date_range(start=start_date, end=end_date, freq='B')


# Asset parameters
S0_initial = np.array([Constants.initialS1, Constants.initialS2])  # Initial prices for Lonza and Sika

# Heston model initial variances
V0_initial = np.array([0.010000, 0.195311])

# Drift terms (assumed to be risk-free rate)
mu = np.array([0.018, 0.018])

# Mean reversion speeds
kappa = np.array([4.483091, 0.901516])

# Long-term variances
theta = np.array([0.124813, 0.184639])

# Volatility of variance
sigma_v = np.array([1, 0.549895])

# Correlation between asset price and its variance
rho = np.array([0.772476, -0.728402])

# Correlation between assets' price processes
rho_assets = np.array([
    [1, 0.4250732997428952],
    [0.4250732997428952, 1]
])

# Simulation parameters
T = 1.25  # Time to maturity (15 months)
N = 314  # Number of time steps (daily steps over 15 months)
nsims = 1000  # Number of simulations

# Define tickers and fetch data
TICKERS = ["LONN.SW", "SIKA.SW"]
START_DATE = Constants.initial_fixing_date
END_DATE = Constants.final_fixing_date

# Fetch historical adjusted close prices
logging.info("Fetching historical asset data...")
data = yf.download(TICKERS, start=START_DATE, end=END_DATE)['Adj Close']

# Handle missing data by forward filling and dropping remaining NaNs
logging.info("Handling missing data...")
data = data.ffill().dropna()

# Ensure data.index is timezone-naive
if data.index.tz is not None:
    logging.info("Converting data.index to timezone-naive.")
    data.index = data.index.tz_convert(None)  # Remove timezone information

# Calculate barrier levels based on initial investment
barrier_levels = {
    "LONN.SW": Constants.barrierS1,
    "SIKA.SW": Constants.barrierS2
}

# Read risk-free rates
logging.info("Reading risk-free rates...")
risk_free_rate_df = pd.read_csv("/Users/guanjuwang/Desktop/TAN /MH4518/rates.csv")

# Check if 'Date' column exists
if 'Date' in risk_free_rate_df.columns:
    # Parse 'Date' with specified format to eliminate UserWarning
    # Replace '%m/%d/%y' with the correct format as per your 'rates.csv'
    try:
        risk_free_rate_df['Date'] = pd.to_datetime(risk_free_rate_df['Date'], format='%m/%d/%y')
    except ValueError as e:
        logging.error(f"Date parsing failed: {e}")
        # Attempt to infer datetime format
        logging.info("Attempting to infer datetime format...")
        risk_free_rate_df['Date'] = pd.to_datetime(risk_free_rate_df['Date'], infer_datetime_format=True,
                                                   errors='coerce')
        # Check for any NaT values after parsing
        if risk_free_rate_df['Date'].isnull().any():
            logging.error(
                "Some dates could not be parsed. Please check your 'rates.csv' for inconsistent date formats.")
            raise ValueError("Date parsing failed due to inconsistent formats.")

    # Set 'Date' as index
    risk_free_rate_df.set_index('Date', inplace=True)

    # Ensure risk_free_rate_df index is timezone-naive
    if risk_free_rate_df.index.tz is not None:
        logging.info("Converting risk_free_rate_df index to timezone-naive.")
        risk_free_rate_df.index = risk_free_rate_df.index.tz_convert(None)  # Remove timezone information

    # Align with data.index
    risk_free_rate_series = risk_free_rate_df['avg/100'].reindex(data.index).ffill().bfill()
else:
    # If 'Date' column does not exist, align by position
    logging.warning("'Date' column not found in rates.csv. Aligning risk-free rates by position.")
    risk_free_rate_series = risk_free_rate_df["avg/100"]

    # Ensure it has enough entries
    if len(risk_free_rate_series) < len(data):
        # Extend the series by repeating the last value
        last_value = risk_free_rate_series.iloc[-1]
        extension = pd.Series([last_value] * (len(data) - len(risk_free_rate_series)))
        risk_free_rate_series = pd.concat([risk_free_rate_series, extension], ignore_index=True)
        logging.warning("Risk-free rate series was extended to match the data length.")
    else:
        # Truncate to match the data length
        risk_free_rate_series = risk_free_rate_series.iloc[:len(data)]


# Define simulation functions (simulate_heston_multidim, gbm_with_emc, etc.)
def simulate_heston_multidim(S0, V0, mu, kappa, theta, sigma_v, rho, rho_assets, T, N, nsims):
    """
    Simulate multidimensional Heston model for multiple assets.

    Returns:
        S: NumPy array of shape (nsims, steps + 1, num_assets)
        V: NumPy array of shape (nsims, steps + 1, num_assets)
        time_grid: NumPy array of time steps
    """
    dt = 1 / 252
    num_assets = len(S0)
    time_grid = np.linspace(0, T, N + 1)

    # Initialize arrays
    S = np.zeros((nsims, N + 1, num_assets))
    V = np.zeros((nsims, N + 1, num_assets))

    # Set initial values
    S[:, 0, :] = S0
    V[:, 0, :] = V0

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
            V_prev = V[:, t - 1, i]
            V_sqrt = np.sqrt(np.maximum(V_prev, 0))
            V[:, t, i] = V_prev + kappa[i] * (theta[i] - V_prev) * dt + sigma_v[i] * V_sqrt * dW_v_i
            V[:, t, i] = np.maximum(V[:, t, i], 0)

            # Simulate asset price
            S_prev = S[:, t - 1, i]
            S[:, t, i] = S_prev * np.exp((mu[i] - 0.5 * V_prev) * dt + V_sqrt * dW_S_i)

    return S, V, time_grid


def get_sliding_window_params(data, risk_free_rate_series, i, total_days=314, window_size=252):
    """
    Extract parameters for the sliding window simulation.
    """
    start_idx = max(0, total_days - window_size - i)
    end_idx = total_days - i
    window_data = data.iloc[start_idx:end_idx]

    # Check if start_idx is within the bounds of risk_free_rate_series
    if start_idx >= len(risk_free_rate_series):
        # Use the last available risk-free rate and log a warning
        window_risk_free_rate = risk_free_rate_series.iloc[-1]
        logging.warning(f"start_idx ({start_idx}) out of bounds. Using last available risk-free rate.")
    else:
        window_risk_free_rate = risk_free_rate_series.iloc[start_idx]

    # Proceed with calculations
    log_returns = np.log(window_data / window_data.shift(1)).dropna()
    drift = log_returns.mean().values
    sigma = log_returns.std().values * np.sqrt(252)
    corr_matrix = log_returns.corr().values

    return drift, sigma, corr_matrix, window_data.iloc[-1].values, window_risk_free_rate


def gbm_with_emc(Nsim, S0, r, sigma, T, steps, corr_matrix):
    dt = T / steps
    L = np.linalg.cholesky(corr_matrix)
    num_assets = len(S0)
    paths = np.zeros((Nsim, steps + 1, num_assets))
    paths[:, 0, :] = S0

    for t in range(1, steps + 1):
        Z = np.random.normal(0, 1, (Nsim, num_assets))
        correlated_Z = Z @ L.T
        Z_i = np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * correlated_Z)
        paths[:, t, :] = paths[:, t - 1, :] * Z_i

    return paths


def gbm_with_antithetic(Msim, S0, r, sigma, T, steps, corr_matrix):
    dt = T / steps
    L = np.linalg.cholesky(corr_matrix)
    num_assets = len(S0)
    paths = np.zeros((2 * Msim, steps + 1, num_assets))
    paths[:, 0, :] = S0  # Set initial prices

    for t in range(1, steps + 1):
        Z = np.random.normal(0, 1, (Msim, num_assets))  # Corrected 'Msims' to 'Msim'
        correlated_Z = Z @ L.T
        Z_i = np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * correlated_Z)

        correlated_Z_antithetic = (-Z) @ L.T
        Z_i_antithetic = np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * correlated_Z_antithetic)
        paths[:Msim, t, :] = paths[:Msim, t - 1, :] * Z_i  # Corrected 'Msims' to 'Msim'
        paths[Msim:, t, :] = paths[Msim:, t - 1, :] * Z_i_antithetic  # Corrected 'Msims' to 'Msim'

    return paths


def gbm_with_control_variate(Nsim, S0, r, sigma, T, steps, corr_matrix):
    primary_paths = gbm_with_emc(Nsim, S0, r, sigma, T, steps, corr_matrix)
    control_paths = gbm_with_emc(Nsim, S0, r, sigma, T, steps, corr_matrix)
    primary_final_prices = np.sum(primary_paths[:, -1], axis=1)
    control_final_prices = np.sum(control_paths[:, -1], axis=1)
    covariance_Y_X = np.cov(primary_final_prices, control_final_prices)[0, 1]
    variance_X = np.var(control_final_prices)
    c_star = -covariance_Y_X / variance_X
    control_variate_adjustment = c_star * (primary_final_prices - control_final_prices)
    adjusted_primary_paths = primary_paths.copy()
    adjusted_primary_paths[:, -1, :] += control_variate_adjustment.reshape(-1, 1)

    return adjusted_primary_paths


def calculate_payoff_vectorized(paths, initial_investment, initpriceS1=Constants.initialS1,
                                initpriceS2=Constants.initialS2,
                                barrierS1=Constants.barrierS1, barrierS2=Constants.barrierS2,
                                conversionratioS1=Constants.conversionratioS1,
                                conversionratioS2=Constants.conversionratioS2):
    """
    Vectorized calculation of payoffs for all simulations.

    Params:
        paths: NumPy array of shape (nsims, steps + 1, num_assets)
        initial_investment: Float, initial investment amount
        initpriceS1, initpriceS2: Floats, initial prices of S1 and S2
        barrierS1, barrierS2: Floats, barrier levels for S1 and S2
        conversionratioS1, conversionratioS2: Floats, conversion ratios for S1 and S2

    Returns:
        payoffs: NumPy array of payoffs for each simulation
    """
    terminalS1 = paths[:, -1, 0]
    terminalS2 = paths[:, -1, 1]

    # Check conditions
    condA = (terminalS1 >= initpriceS1) & (terminalS2 >= initpriceS2)
    condB = (paths[:, :, 0].min(axis=1) >= barrierS1) & (paths[:, :, 1].min(axis=1) >= barrierS2)

    # Initialize payoffs
    payoffs = np.zeros(paths.shape[0])

    # Case 1: condA or condB
    case1 = condA | condB
    payoffs[case1] = (1 + (0.08 / 12 * 15)) * initial_investment + initial_investment

    # Case 2: Not condA and not condB
    case2 = ~condA & ~condB
    # Determine which asset to use for conversion
    use_S1 = terminalS1 <= terminalS2
    conversion_ratios = np.where(use_S1, conversionratioS1, conversionratioS2)
    # Apply conversion ratios only to case2 indices
    payoffs[case2] = (1 + (0.08 / 12 * 15)) * initial_investment + initial_investment * conversion_ratios[case2]

    # Case 3: Any asset reaches zero
    zero_S1 = (paths[:, :, 0] == 0).any(axis=1)
    zero_S2 = (paths[:, :, 1] == 0).any(axis=1)
    case3 = zero_S1 | zero_S2
    payoffs[case3] = (1 + (0.08 / 12 * 15)) * initial_investment

    return payoffs


def monte_carlo_simulation_with_sliding_window_with_ci(S0, data, total_days, window_size, Nsim, Msim,
                                                       sliding_window_offset=0):
    """
    Perform Monte Carlo simulations over sliding windows using different simulation methods,
    and calculate confidence intervals.

    Params:
        S0: Initial asset prices (list or NumPy array)
        data: Historical asset data (DataFrame)
        total_days: Total number of days in the simulation
        window_size: Size of each sliding window
        Nsim: Number of simulations for GBM and Heston
        Msim: Number of simulations for Antithetic Variate
        sliding_window_offset: Offset for sliding windows (default is 0)

    Returns:
        Dictionary containing mean payoffs and standard deviations for each simulation method
    """
    sliding_windows_payoffs = {
        'gbm': [],
        'antithetic': [],
        'control_variate': [],
        'heston': []  # New entry for Heston
    }

    sliding_windows_std = {
        'gbm': [],
        'antithetic': [],
        'control_variate': [],
        'heston': []
    }

    num_windows = total_days - window_size + 1

    for i in range(num_windows):
        drift, sigma, corr_matrix, S0_window, r_window = get_sliding_window_params(
            data, risk_free_rate_series, i, total_days, window_size
        )
        T = window_size / 252

        # GBM with EMC
        paths_gbm = gbm_with_emc(Nsim, S0_window, r_window, sigma, T, window_size, corr_matrix)
        payoffs_gbm = calculate_payoff_vectorized(paths_gbm, Constants.initial_investment)
        mean_payoff_gbm = np.mean(payoffs_gbm)
        std_payoff_gbm = np.std(payoffs_gbm)
        sliding_windows_payoffs['gbm'].append(mean_payoff_gbm)
        sliding_windows_std['gbm'].append(std_payoff_gbm)

        # GBM with Antithetic
        paths_an = gbm_with_antithetic(Msim, S0_window, r_window, sigma, T, window_size, corr_matrix)
        payoffs_an = calculate_payoff_vectorized(paths_an, Constants.initial_investment)
        mean_payoff_an = np.mean(payoffs_an)
        std_payoff_an = np.std(payoffs_an)
        sliding_windows_payoffs['antithetic'].append(mean_payoff_an)
        sliding_windows_std['antithetic'].append(std_payoff_an)

        # GBM with Control Variate
        paths_cv = gbm_with_control_variate(Nsim, S0_window, r_window, sigma, T, window_size, corr_matrix)
        payoffs_cv = calculate_payoff_vectorized(paths_cv, Constants.initial_investment)
        mean_payoff_cv = np.mean(payoffs_cv)
        std_payoff_cv = np.std(payoffs_cv)
        sliding_windows_payoffs['control_variate'].append(mean_payoff_cv)
        sliding_windows_std['control_variate'].append(std_payoff_cv)

        # Heston Model
        S_heston, V_heston, time_grid_heston = simulate_heston_multidim(
            S0=S0_window,
            V0=V0_initial,
            mu=mu,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            rho_assets=rho_assets,
            T=T,
            N=window_size,
            nsims=Nsim
        )

        # Compute payoffs for Heston paths
        payoffs_heston = calculate_payoff_vectorized(S_heston, Constants.initial_investment)
        mean_payoff_heston = np.mean(payoffs_heston)
        std_payoff_heston = np.std(payoffs_heston)
        sliding_windows_payoffs['heston'].append(mean_payoff_heston)
        sliding_windows_std['heston'].append(std_payoff_heston)

        # Log progress every 50 windows and at the end
        if (i + 1) % 50 == 0 or i == num_windows - 1:
            logging.info(f"Completed {i + 1}/{num_windows} sliding windows.")

    # Convert lists to NumPy arrays for plotting
    payoffs_gbm_all = np.array(sliding_windows_payoffs['gbm'])
    payoffs_an_all = np.array(sliding_windows_payoffs['antithetic'])
    payoffs_cv_all = np.array(sliding_windows_payoffs['control_variate'])
    payoffs_heston_all = np.array(sliding_windows_payoffs['heston'])

    std_gbm_all = np.array(sliding_windows_std['gbm'])
    std_an_all = np.array(sliding_windows_std['antithetic'])
    std_cv_all = np.array(sliding_windows_std['control_variate'])
    std_heston_all = np.array(sliding_windows_std['heston'])

    return {
        'payoffs_gbm': payoffs_gbm_all,
        'std_gbm': std_gbm_all,
        'payoffs_an': payoffs_an_all,
        'std_an': std_an_all,
        'payoffs_cv': payoffs_cv_all,
        'std_cv': std_cv_all,
        'payoffs_heston': payoffs_heston_all,
        'std_heston': std_heston_all
    }


# Run the simulation with confidence intervals
logging.info("Starting Monte Carlo simulation with sliding windows...")
try:
    simulation_results_ci = monte_carlo_simulation_with_sliding_window_with_ci(
        S0=S0_initial,
        data=data,
        total_days=len(data),
        window_size=252,  # 1 year trading days
        Nsim=10000,  # Number of simulations for GBM and Heston
        Msim=5000  # Number of simulations for Antithetic Variate
    )
except Exception as e:
    logging.error(f"An error occurred during simulation: {e}")
    raise

# Extract mean payoffs and standard deviations
payoffs_gbm = simulation_results_ci['payoffs_gbm']
std_gbm = simulation_results_ci['std_gbm']
payoffs_an = simulation_results_ci['payoffs_an']
std_an = simulation_results_ci['std_an']
payoffs_cv = simulation_results_ci['payoffs_cv']
std_cv = simulation_results_ci['std_cv']
payoffs_heston = simulation_results_ci['payoffs_heston']
std_heston = simulation_results_ci['std_heston']

# Generate window start dates
window_start_dates = Dates.get_list_dates(Constants.initial_fixing_date, Constants.final_fixing_date)
window_start_dates = pd.Series(window_start_dates[:len(payoffs_gbm)]).reset_index(drop=True)

# Define confidence multiplier for 95% confidence interval
confidence_multiplier = 1.96

# Calculate confidence intervals
ci_gbm_upper = payoffs_gbm + confidence_multiplier * (std_gbm / np.sqrt(10000))
ci_gbm_lower = payoffs_gbm - confidence_multiplier * (std_gbm / np.sqrt(10000))

ci_an_upper = payoffs_an + confidence_multiplier * (std_an / np.sqrt(5000))
ci_an_lower = payoffs_an - confidence_multiplier * (std_an / np.sqrt(5000))

ci_cv_upper = payoffs_cv + confidence_multiplier * (std_cv / np.sqrt(10000))
ci_cv_lower = payoffs_cv - confidence_multiplier * (std_cv / np.sqrt(10000))

ci_heston_upper = payoffs_heston + confidence_multiplier * (std_heston / np.sqrt(10000))
ci_heston_lower = payoffs_heston - confidence_multiplier * (std_heston / np.sqrt(10000))

# Plotting Expected Payoff Over Sliding Windows with Confidence Intervals
plt.figure(figsize=(14, 7))

# GBM
plt.plot(window_start_dates, payoffs_gbm, label='GBM', color='blue')
plt.fill_between(window_start_dates, ci_gbm_lower, ci_gbm_upper, color='blue', alpha=0.2)

# Antithetic Variate
plt.plot(window_start_dates, payoffs_an, label='Antithetic', color='green')
plt.fill_between(window_start_dates, ci_an_lower, ci_an_upper, color='green', alpha=0.2)

# Control Variate
plt.plot(window_start_dates, payoffs_cv, label='Control Variate', color='red')
plt.fill_between(window_start_dates, ci_cv_lower, ci_cv_upper, color='red', alpha=0.2)

# Heston Model
plt.plot(window_start_dates, payoffs_heston, label='Heston', color='purple')
plt.fill_between(window_start_dates, ci_heston_lower, ci_heston_upper, color='purple', alpha=0.2)

plt.title('Expected Payoff Over Sliding Windows with 95% Confidence Intervals')
plt.xlabel('Sliding Window Start Date')
plt.ylabel('Expected Payoff (CHF)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

