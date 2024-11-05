import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

# ===========================
# 1. Define Utility Functions
# ===========================

def get_early_observation_dates(start_date, end_date, frequency='3M'):
    """
    Generate early observation dates based on frequency.

    Parameters:
    - start_date: pd.Timestamp, start date of the simulation.
    - end_date: pd.Timestamp, end date of the simulation.
    - frequency: str, frequency string for date_range (default '3M' for quarterly).

    Returns:
    - pd.DatetimeIndex of observation dates.
    """
    # Early observations start after 6 months
    start_observation = start_date + relativedelta(months=6)
    return pd.date_range(start=start_observation, end=end_date, freq=frequency)

def add_business_days(date, days):
    """
    Add a specified number of business days to a date.

    Parameters:
    - date: pd.Timestamp, the original date.
    - days: int, number of business days to add.

    Returns:
    - pd.Timestamp, the new date after adding business days.
    """
    return pd.bdate_range(start=date, periods=days+1)[-1]

# ===========================
# 2. Define Constants and Classes
# ===========================

# Asset parameters
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
N = 315       # Number of time steps (daily steps over 15 months)
nsims = 1000  # Number of simulations

# Initialize the 'cs' class with required attributes
class cs:
    initial_fixing_date = pd.Timestamp('2023-04-30')
    final_fixing_date = pd.Timestamp('2024-07-30')
    initialS1 = S0[0]
    initialS2 = S0[1]
    barrierS1 = 0.6 * S0[0]  # 60% barrier level
    barrierS2 = 0.6 * S0[1]
    ticker_list = ['Lonza', 'Sika']
    num_ticker = 2

# ===========================
# 3. Define Simulation Function
# ===========================

def simulate_heston_multidim(S0, V0, mu, kappa, theta, sigma_v, rho, rho_assets, T, N, nsims):
    """
    Simulate multidimensional Heston model for multiple assets.
    """
    dt = T / N
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

    # Combined correlation matrix
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

# ===========================
# 4. Define Payoff Functions
# ===========================

def calculate_payoff(pathS1, pathS2, params, fdos, dates):
    '''
    Calculates the payoff and payoff date for each simulation based on the product conditions.

    Returns:
        payoffs: Array of payoffs for each simulation
        payoff_dates: Series of payoff dates for each simulation
    '''
    initinvestment = params['Denomination']
    initpriceS1 = params.get('initpriceS1', 549.60)
    initpriceS2 = params.get('initpriceS2', 240.40)
    barrierS1 = params.get('barrierS1', 329.76)
    barrierS2 = params.get('barrierS2', 144.24)
    conversionratioS1 = params.get('conversionratioS1', 1.8195)
    conversionratioS2 = params.get('conversionratioS2', 4.1597)
    coupon_rate = params.get('Coupon_Rate', 0.08 / 12)  # Monthly coupon rate

    payoffs = np.zeros(len(pathS1.columns))
    # Initialize payoff_dates as a pandas Series with NaT values
    payoff_dates = pd.Series(pd.NaT, index=pathS1.columns, dtype='datetime64[ns]')

    for sim in pathS1.columns:
        path_lonza = pathS1[sim]
        path_sika = pathS2[sim]
        terminalS1 = path_lonza.iloc[-1]
        terminalS2 = path_sika.iloc[-1]

        # Initialize variables
        early_redeem = False
        payoff = 0.0
        payoff_date = params['final_redemption_date']

        # Check early redemption on observation dates
        for obs_date in params['early_observation_dates']:
            # Ensure obs_date is within the simulation period
            if obs_date not in path_lonza.index:
                continue
            if (path_lonza.loc[obs_date] >= initpriceS1) and (path_sika.loc[obs_date] >= initpriceS2):
                # Early redemption triggered
                payoff_date = pd.Timestamp(obs_date)
                days = (payoff_date - fdos).days
                periods = days / 30  # Approximate number of months
                payoff = initinvestment * (1 + coupon_rate * periods)
                early_redeem = True
                break

        if not early_redeem:
            # No early redemption, check barrier and terminal conditions
            if (path_lonza.min() >= barrierS1) and (path_sika.min() >= barrierS2) and (terminalS1 >= initpriceS1) and (terminalS2 >= initpriceS2):
                # Best case: receive principal and accrued coupon
                days = (payoff_date - fdos).days
                periods = days / 30  # Approximate number of months
                payoff = initinvestment * (1 + coupon_rate * periods)
            elif (path_lonza.min() < barrierS1) or (path_sika.min() < barrierS2):
                # Barrier breached: convert to worst-performing asset
                worst_perf = min(terminalS1 / initpriceS1, terminalS2 / initpriceS2)
                if worst_perf < 1.0:
                    if worst_perf == terminalS1 / initpriceS1:
                        conversion_ratio = conversionratioS1
                        converted_amount = terminalS1 * conversionratioS1
                    else:
                        conversion_ratio = conversionratioS2
                        converted_amount = terminalS2 * conversionratioS2
                    days = (payoff_date - fdos).days
                    periods = days / 30  # Approximate number of months
                    payoff = (initinvestment * (1 + coupon_rate * periods)) + converted_amount
                else:
                    # No conversion, receive principal and accrued coupon
                    days = (payoff_date - fdos).days
                    periods = days / 30  # Approximate number of months
                    payoff = initinvestment * (1 + coupon_rate * periods)
            else:
                # Any Reference Share drops to zero
                days = (payoff_date - fdos).days
                periods = days / 30  # Approximate number of months
                payoff = initinvestment * (1 + coupon_rate * periods)

        payoffs[sim] = payoff
        payoff_dates[sim] = payoff_date

    return payoffs, payoff_dates

def payoff(paths1, paths2, params, fdos, dates):
    """
    Calculates the payoff and payoff date for each simulation based on the product conditions.

    Params:
        paths1: DataFrame of simulated paths for Lonza (columns: simulations)
        paths2: DataFrame of simulated paths for Sika (columns: simulations)
        params: Dictionary containing product parameters
        fdos: First date of simulation (pricing date)
        dates: DatetimeIndex of simulation dates

    Returns:
        payoffs: Array of payoffs for each simulation
        payoff_dates: Series of payoff dates for each simulation
    """
    payoffs, payoff_dates = calculate_payoff(paths1, paths2, params, fdos, dates)
    return payoffs, payoff_dates

# ===========================
# 5. Define Discount Function
# ===========================

def discount_factor(payoff_date, fdos, r):
    """
    Calculate the discount factor from payoff_date to fdos.

    Parameters:
        payoff_date (pd.Series): Series of payoff dates.
        fdos (pd.Timestamp): Pricing date.
        r (float): Annual risk-free rate.

    Returns:
        pd.Series: Discount factors for each payoff date.
    """
    # Calculate the difference in days using .dt.days
    delta = (payoff_date - fdos).dt.days
    T = delta / 365.0  # Time in years
    discount = np.exp(-r * T)
    return discount

# ===========================
# 6. Define Expected Price Calculation
# ===========================

def calculate_expected_daily_price(payoff_dates, payoffs, r, fdos, dates):
    """
    Calculate the expected daily price of the financial product.

    Parameters:
        payoff_dates (pd.Series): Series of payoff dates for each simulation.
        payoffs (np.ndarray): Array of payoffs for each simulation.
        r (float): Annual risk-free rate.
        fdos (pd.Timestamp): Pricing date.
        dates (pd.DatetimeIndex): All simulation dates.

    Returns:
        pd.Series: Expected price for each day.
    """
    expected_prices = []

    for current_date in dates:
        # Identify payoffs that occur on or after the current_date
        mask = payoff_dates >= current_date
        relevant_payoffs = payoffs[mask]
        relevant_dates = payoff_dates[mask]

        if relevant_dates.empty:
            # No future payoffs; expected price is 0
            expected_prices.append(0.0)
            continue

        # Calculate the difference in days between payoff_dates and current_date
        delta_payoff = (relevant_dates - current_date).dt.days

        # Convert days to years
        T_discount = delta_payoff / 365.0

        # Calculate discount factors
        discounts = np.exp(-r * T_discount)

        # Discount the relevant payoffs to current_date
        discounted = relevant_payoffs * discounts

        # Expected price is the average of discounted payoffs
        expected_price = np.mean(discounted)
        expected_prices.append(expected_price)

    return pd.Series(expected_prices, index=dates)


# ===========================
# 7. Execute Simulation and Calculations
# ===========================

# Define first date of simulation
fdos = cs.initial_fixing_date

# Create date range
dates = pd.bdate_range(start=fdos, periods=N+1)  # Business days

# Simulate Heston model paths
S_sim, V_sim, time_grid = simulate_heston_multidim(
    S0, V0, mu, kappa, theta, sigma_v, rho, rho_assets, T, N, nsims)

# Convert simulated prices to DataFrames
lonza_paths = pd.DataFrame(S_sim[0].T, index=dates)
sika_paths = pd.DataFrame(S_sim[1].T, index=dates)

# Define product parameters
params_product = {
    'Denomination': 1000,            # Denomination of the product
    'Coupon_Rate': (0.08 / 12),      # Monthly coupon rate
    'Settlement_Delay': 2,           # Settlement delay in days
    'initpriceS1': 549.60,           # Initial price of Lonza
    'initpriceS2': 240.40,           # Initial price of Sika
    'barrierS1': 329.76,             # Barrier level for Lonza
    'barrierS2': 144.24,             # Barrier level for Sika
    'conversionratioS1': 1.8195,     # Conversion ratio for Lonza
    'conversionratioS2': 4.1597,     # Conversion ratio for Sika
    'final_redemption_date': dates[-1]  # Final redemption date
}

# Define early observation dates (quarterly after 6 months)
early_observation_dates = get_early_observation_dates(
    start_date=fdos,
    end_date=params_product['final_redemption_date'],
    frequency='3M'
)

# Update params with observation dates and final redemption date
params_product.update({
    'early_observation_dates': early_observation_dates
})

# Calculate payoffs and payoff dates
payoffs, payoff_dates = payoff(lonza_paths, sika_paths, params_product, fdos, dates)

# Define risk-free rate for discounting
risk_free_rate = 0.018  # 1.8% annual

# Calculate discount factors for all payoffs
discount_factors = discount_factor(payoff_dates, fdos, risk_free_rate)

# Calculate discounted payoffs (Present Value as of fdos)
discounted_payoffs = payoffs * discount_factors

# Calculate expected daily price
expected_daily_price = calculate_expected_daily_price(
    payoff_dates,
    payoffs,
    risk_free_rate,
    fdos,
    dates
)

# ===========================
# 8. Plotting Results
# ===========================

# Plot simulated stock price paths for Lonza
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(lonza_paths.index, lonza_paths.iloc[:, i], label=f'Simulation {i+1}')
plt.title('Simulated Stock Price Paths for Lonza')
plt.xlabel('Date')
plt.ylabel('Price (CHF)')
plt.legend()
plt.grid(True)
plt.show()

# Plot simulated stock price paths for Sika
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(sika_paths.index, sika_paths.iloc[:, i], label=f'Simulation {i+1}')
plt.title('Simulated Stock Price Paths for Sika')
plt.xlabel('Date')
plt.ylabel('Price (CHF)')
plt.legend()
plt.grid(True)
plt.show()

# Plot distribution of payoffs
plt.figure(figsize=(10, 6))
plt.hist(payoffs, bins=50, edgecolor='k', alpha=0.7)
plt.title('Distribution of Structured Product Payoffs')
plt.xlabel('Payoff (CHF)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot Payoff vs. Worst Asset Performance
# Extract terminal prices
lonza_terminal = lonza_paths.iloc[-1, :].values
sika_terminal = sika_paths.iloc[-1, :].values

# Worst performance
worst_perf = np.minimum(lonza_terminal / cs.initialS1, sika_terminal / cs.initialS2)

plt.figure(figsize=(10, 6))
plt.scatter(worst_perf, payoffs, alpha=0.5)
plt.title('Payoff vs. Worst Asset Performance')
plt.xlabel('Worst Asset Performance')
plt.ylabel('Payoff (CHF)')
plt.grid(True)
plt.show()

# Plot Expected Daily Price
plt.figure(figsize=(12, 6))
plt.plot(expected_daily_price.index, expected_daily_price.values, label='Expected Daily Price', color='blue')
plt.title('Expected Daily Price of Autocallable Barrier Reverse Convertible')
plt.xlabel('Date')
plt.ylabel('Expected Price (CHF)')
plt.legend()
plt.grid(True)
plt.show()

print (expected_daily_price)