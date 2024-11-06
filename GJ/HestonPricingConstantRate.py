import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set a random seed for reproducibility
np.random.seed(42)

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
    fixed_interest_rate = 0.018  # 1.8%


# Define Dates functions
class Dates:
    @staticmethod
    def get_list_dates(start_date, end_date):
        """
        Generate a list of business dates between start_date and end_date.
        """
        return pd.date_range(start=start_date, end=end_date, freq='B')  # Business days


# Asset parameters
S0_initial = np.array([Constants.initialS1, Constants.initialS2])  # Initial prices for Lonza and Sika

# Heston model initial variances
V0_initial = np.array([0.010000, 0.195311])

# Drift terms (assumed to be risk-free rate)
mu = np.array([Constants.fixed_interest_rate, Constants.fixed_interest_rate])

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


# Define Heston Model Simulation Function
def simulate_heston_multidim(S0, V0, mu, kappa, theta, sigma_v, rho, rho_assets, T, N, nsims):
    """
    Simulate multidimensional Heston model for multiple assets.

    Returns:
        S: NumPy array of shape (nsims, steps + 1, num_assets)
        V: NumPy array of shape (nsims, steps + 1, num_assets)
        time_grid: NumPy array of time steps
    """
    dt = 1 / 252  # Daily steps
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
            V[:, t, i] = np.maximum(V[:, t, i], 0)  # Prevent negative variances

            # Simulate asset price
            S_prev = S[:, t - 1, i]
            S[:, t, i] = S_prev * np.exp((mu[i] - 0.5 * V_prev) * dt + V_sqrt * dW_S_i)

    return S, V, time_grid


def get_sliding_window_params_constant_rate(data, i, window_size=252, fixed_rate=Constants.fixed_interest_rate):
    """
    Extract parameters for the sliding window simulation with a constant interest rate.

    Params:
        data: Historical asset data (DataFrame)
        i: Current window index
        window_size: Size of each sliding window
        fixed_rate: Constant interest rate (e.g., 0.018)

    Returns:
        drift: Drift coefficients (unused in this context)
        sigma: Volatility coefficients
        corr_matrix: Correlation matrix of log returns
        S0_window: Initial asset prices at the end of the window
        r_window: Fixed interest rate
    """
    start_idx = i
    end_idx = i + window_size
    window_data = data.iloc[start_idx:end_idx]

    # Set fixed interest rate
    r_window = fixed_rate

    # Proceed with calculations
    log_returns = np.log(window_data / window_data.shift(1)).dropna()
    drift = log_returns.mean().values  # Could be set to fixed rate if desired
    sigma = log_returns.std().values * np.sqrt(252)
    corr_matrix = log_returns.corr().values

    return drift, sigma, corr_matrix, window_data.iloc[-1].values, r_window


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

    # Define the interest factor correctly
    interest_factor = (1 + (Constants.fixed_interest_rate / 12 * 15))  # 1 + (0.018 / 12 * 15) = 1.0225

    # Case 1: condA or condB
    case1 = condA | condB
    payoffs[case1] = interest_factor * initial_investment  # 1.0225 * 1000 = 1022.5 CHF

    # Case 2: Not condA and not condB
    case2 = ~condA & ~condB
    # Determine which asset to use for conversion
    use_S1 = terminalS1 <= terminalS2

    # Calculate number of shares initially held
    num_shares_S1 = initial_investment / initpriceS1
    num_shares_S2 = initial_investment / initpriceS2

    # Apply conversion ratios to the number of shares
    num_converted_S1 = num_shares_S1 * conversionratioS1
    num_converted_S2 = num_shares_S2 * conversionratioS2

    # Select conversion ratios based on which asset to use
    num_converted = np.where(use_S1, num_converted_S1, num_converted_S2)

    # Calculate the converted investment value at terminal prices
    converted_value = np.where(use_S1, num_converted * terminalS1, num_converted * terminalS2)

    # Assign payoffs for Case 2 (ONLY the converted value, not adding initial investment)
    payoffs[case2] = converted_value[case2]

    # Case 3: Any asset reaches zero
    zero_S1 = (paths[:, :, 0] == 0).any(axis=1)
    zero_S2 = (paths[:, :, 1] == 0).any(axis=1)
    case3 = zero_S1 | zero_S2
    payoffs[case3] = interest_factor * initial_investment  # 1022.5 CHF

    return payoffs


def monte_carlo_simulation_with_sliding_window_heston_constant_rate(S0, data, total_days, window_size, Nsim,
                                                                    fixed_rate=Constants.fixed_interest_rate):
    """
    Perform Monte Carlo simulations over sliding windows using only the Heston model with a constant interest rate,
    and calculate confidence intervals.

    Params:
        S0: Initial asset prices (list or NumPy array)
        data: Historical asset data (DataFrame)
        total_days: Total number of days in the simulation
        window_size: Size of each sliding window
        Nsim: Number of simulations for Heston
        fixed_rate: Constant interest rate (e.g., 0.018)

    Returns:
        Dictionary containing mean payoffs and standard deviations for the Heston model
    """
    sliding_windows_payoffs = {
        'heston': []  # Only Heston
    }

    sliding_windows_std = {
        'heston': []
    }

    num_windows = total_days - window_size + 1

    for i in range(num_windows):
        drift, sigma, corr_matrix, S0_window, r_window = get_sliding_window_params_constant_rate(
            data, i, window_size, fixed_rate
        )
        T = window_size / 252  # Adjusted to window_size

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
    payoffs_heston_all = np.array(sliding_windows_payoffs['heston'])
    std_heston_all = np.array(sliding_windows_std['heston'])

    return {
        'payoffs_heston': payoffs_heston_all,
        'std_heston': std_heston_all
    }

# Run the simulation with confidence intervals
logging.info("Starting Monte Carlo simulation with sliding windows (Heston Model Only, Constant Rate)...")
try:
    simulation_results_ci = monte_carlo_simulation_with_sliding_window_heston_constant_rate(
        S0=S0_initial,
        data=data,
        total_days=len(data),
        window_size=252,  # 1 year trading days
        Nsim=10000,  # Number of simulations for Heston
        fixed_rate=Constants.fixed_interest_rate  # Constant interest rate
    )
except Exception as e:
    logging.error(f"An error occurred during simulation: {e}")
    raise

# Extract mean payoffs and standard deviations
payoffs_heston = simulation_results_ci['payoffs_heston']
std_heston = simulation_results_ci['std_heston']

# Generate window start dates
window_start_dates = Dates.get_list_dates(Constants.initial_fixing_date, Constants.final_fixing_date)
window_start_dates = pd.Series(window_start_dates[:len(payoffs_heston)]).reset_index(drop=True)

# Define confidence multiplier for 95% confidence interval
confidence_multiplier = 1.96

# Calculate confidence intervals
ci_heston_upper = payoffs_heston + confidence_multiplier * (std_heston / np.sqrt(10000))
ci_heston_lower = payoffs_heston - confidence_multiplier * (std_heston / np.sqrt(10000))

# Plotting Expected Payoff Over Sliding Windows with Confidence Intervals (Heston Model Only)
plt.figure(figsize=(14, 7))

# Heston Model
plt.plot(window_start_dates, payoffs_heston, label='Heston', color='purple')
plt.fill_between(window_start_dates, ci_heston_lower, ci_heston_upper, color='purple', alpha=0.2)

plt.title('Expected Payoff Over Sliding Windows with 95% Confidence Intervals (Heston Model)')
plt.xlabel('Sliding Window Start Date')
plt.ylabel('Expected Payoff (CHF)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
