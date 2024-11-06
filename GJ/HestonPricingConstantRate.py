import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import logging
from pandas.tseries.offsets import BDay
from dateutil.relativedelta import relativedelta

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
    barrierS1 = 329.76  # 60% of 549.60
    barrierS2 = 144.24  # 60% of 240.40
    conversionratioS1 = 1.8195
    conversionratioS2 = 4.1597
    initial_investment = 1000  # CHF
    fixed_interest_rate = 0.018  # 8% annual coupon

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

# Product parameters
params = {
    'Denomination': 1000,            # Denomination of the product
    'Coupon_Rate': 0.02,             # Quarterly coupon rate (8% p.a. / 4)
    'Settlement_Delay': 2            # Settlement delay in business days
}

# Simulation parameters
T = 1.25  # Time to maturity (15 months)
N = 314  # Number of time steps (daily steps over 15 months)
nsims = 10000  # Number of simulations (increased for better accuracy)

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
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        logging.error("Cholesky decomposition failed. Check the correlation matrix for positive definiteness.")
        raise

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

def get_early_observation_dates(initial_fixing_date, final_fixing_date, frequency='quarterly'):
    """
    Generate early redemption observation dates, quarterly starting after 6 months.
    """
    initial_date = pd.to_datetime(initial_fixing_date)
    final_date = pd.to_datetime(final_fixing_date)
    first_observation = initial_date + relativedelta(months=6)
    if frequency == 'quarterly':
        freq = 'QS'  # Quarter start frequency
    elif frequency == 'monthly':
        freq = 'MS'  # Month start frequency
    else:
        freq = 'A'   # Annual frequency
    observation_dates = pd.date_range(start=first_observation, end=final_date, freq=freq)
    # Adjust to business days if necessary
    observation_dates = observation_dates[
        observation_dates.isin(pd.bdate_range(start=initial_fixing_date, end=final_fixing_date))
    ]
    return observation_dates

def add_business_days(start_date, num_days):
    """
    Add a specified number of business days to a date.
    """
    start = pd.to_datetime(start_date)
    end = start + BDay(num_days)
    return end

def get_sliding_window_params_constant_rate(data, i, window_size=252, fixed_rate=Constants.fixed_interest_rate):
    """
    Extract parameters for the sliding window simulation with a constant interest rate.

    Params:
        data: Historical asset data (DataFrame)
        i: Current window index
        window_size: Size of each sliding window
        fixed_rate: Constant interest rate (e.g., 0.08 for 8%)

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


def payoff(paths1, paths2, params, fdos):
    """
    Calculate payoffs using the corrected conversion ratio application.

    Params:
        paths1: DataFrame of simulated paths for Lonza (columns: simulations)
        paths2: DataFrame of simulated paths for Sika (columns: simulations)
        params: Dictionary containing product parameters
        fdos: First date of simulation (pricing date)

    Returns:
        payoffs: Array of payoffs for each simulation
    """

    def get_number_of_coupon_periods(start_date, end_date, frequency='quarterly'):
        delta = relativedelta(end_date, start_date)
        if frequency == 'quarterly':
            periods = delta.years * 4 + delta.months // 3
        elif frequency == 'monthly':
            periods = delta.years * 12 + delta.months
        else:
            periods = delta.years  # Assuming annual
        return periods

    def checkbarrier(lonza, sika, barrierS1, barrierS2):
        # Assuming daily monitoring
        barrier_breach = ((lonza <= barrierS1) | (sika <= barrierS2)).any()
        return barrier_breach

    def check_terminal(lonza, sika, initialS1, initialS2):
        terminal_condition = (lonza.iloc[-1] < initialS1) or (sika.iloc[-1] < initialS2)
        return terminal_condition

    # Ensure columns are sorted if necessary
    paths1 = paths1.reindex(sorted(paths1.columns), axis=1)
    paths2 = paths2.reindex(sorted(paths2.columns), axis=1)
    simulations = paths1.columns  # List of simulation names

    # Generate Early Redemption Observation Dates
    early_observation_dates = get_early_observation_dates(Constants.initial_fixing_date, Constants.final_fixing_date)

    payoffs = []

    for sim_name in simulations:
        early_redeem = False
        lonza = paths1[sim_name]
        sika = paths2[sim_name]

        # Iterate through Early Redemption Observation Dates
        for t_date in early_observation_dates:
            if t_date not in lonza.index:
                continue  # Skip if the date is not in the simulation (could happen due to holidays)
            if (lonza.loc[t_date] >= Constants.initialS1) and (sika.loc[t_date] >= Constants.initialS2):
                # Early redemption
                early_redemption_date = add_business_days(t_date, params.get('Settlement_Delay', 2))
                periods = get_number_of_coupon_periods(
                    pd.to_datetime(fdos),
                    pd.to_datetime(early_redemption_date),
                    frequency='quarterly'
                )
                payoff_value = params['Denomination'] * (1 + params['Coupon_Rate'] * periods)
                payoffs.append(payoff_value)
                early_redeem = True
                break  # Exit the early observation loop

        if not early_redeem:
            barrierhit = checkbarrier(lonza, sika, Constants.barrierS1, Constants.barrierS2)
            terminallower = check_terminal(lonza, sika, Constants.initialS1, Constants.initialS2)

            if not barrierhit and not terminallower:  # Best case scenario
                periods = get_number_of_coupon_periods(
                    pd.to_datetime(Constants.initial_fixing_date),
                    pd.to_datetime(Constants.final_fixing_date),
                    frequency='quarterly'
                )
                payoff_value = params['Denomination'] * (1 + params['Coupon_Rate'] * periods)
                payoffs.append(payoff_value)
            elif not barrierhit and terminallower:
                # At least one asset below initial level but no barrier breached
                periods = get_number_of_coupon_periods(
                    pd.to_datetime(Constants.initial_fixing_date),
                    pd.to_datetime(Constants.final_fixing_date),
                    frequency='quarterly'
                )
                # Determine worse performing asset
                perf_lonza = lonza.iloc[-1] / Constants.initialS1
                perf_sika = sika.iloc[-1] / Constants.initialS2
                if perf_lonza < perf_sika:
                    conversionratio = Constants.conversionratioS1
                else:
                    conversionratio = Constants.conversionratioS2
                # Calculate converted value correctly
                converted_value = params['Denomination'] * (conversionratio / 100)
                payoff_value = params['Denomination'] * (1 + params['Coupon_Rate'] * periods) + converted_value
                payoffs.append(payoff_value)
            elif barrierhit and terminallower:
                # At least one barrier breached and at least one asset below initial level
                periods = get_number_of_coupon_periods(
                    pd.to_datetime(Constants.initial_fixing_date),
                    pd.to_datetime(Constants.final_fixing_date),
                    frequency='quarterly'
                )
                # Determine worse performing asset
                perf_lonza = lonza.iloc[-1] / Constants.initialS1
                perf_sika = sika.iloc[-1] / Constants.initialS2
                if perf_lonza < perf_sika:
                    conversionratio = Constants.conversionratioS1
                else:
                    conversionratio = Constants.conversionratioS2
                # Calculate converted value correctly
                converted_value = params['Denomination'] * (conversionratio / 100)
                payoff_value = params['Denomination'] * (1 + params['Coupon_Rate'] * periods) + converted_value
                payoffs.append(payoff_value)
            elif terminallower and (lonza.iloc[-1] == 0 or sika.iloc[-1] == 0):
                # Any asset drops to zero (worst case)
                periods = get_number_of_coupon_periods(
                    pd.to_datetime(Constants.initial_fixing_date),
                    pd.to_datetime(Constants.final_fixing_date),
                    frequency='quarterly'
                )
                payoff_value = params['Denomination'] * (1 + params['Coupon_Rate'] * periods)
                payoffs.append(payoff_value)
            else:
                # Default case, return coupons only
                periods = get_number_of_coupon_periods(
                    pd.to_datetime(Constants.initial_fixing_date),
                    pd.to_datetime(Constants.final_fixing_date),
                    frequency='quarterly'
                )
                payoff_value = params['Denomination'] * (1 + params['Coupon_Rate'] * periods)
                payoffs.append(payoff_value)

    payoffs = np.array(payoffs)
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
        fixed_rate: Constant interest rate (e.g., 0.08 for 8%)

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

        # Convert simulation paths to DataFrame format
        df_S1 = pd.DataFrame(S_heston[:, :, 0].T, columns=[f"sim_{j}" for j in range(Nsim)])
        df_S2 = pd.DataFrame(S_heston[:, :, 1].T, columns=[f"sim_{j}" for j in range(Nsim)])

        # Assign proper date indices
        simulation_dates = Dates.get_list_dates(Constants.initial_fixing_date, Constants.final_fixing_date)
        simulation_dates = simulation_dates[:window_size + 1]  # Ensure correct length
        df_S1.index = simulation_dates
        df_S2.index = simulation_dates

        # Calculate payoffs using the provided payoff function
        fdos = pd.to_datetime(Constants.initial_fixing_date)
        payoffs_heston = payoff(df_S1, df_S2, params, fdos)
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
        Nsim=nsims,  # Number of simulations for Heston
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
ci_heston_upper = payoffs_heston + confidence_multiplier * (std_heston / np.sqrt(nsims))
ci_heston_lower = payoffs_heston - confidence_multiplier * (std_heston / np.sqrt(nsims))

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

