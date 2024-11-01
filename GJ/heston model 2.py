import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Asset parameters
S0 = np.array([549.60, 240.40])  # Initial prices for Lonza and Sika
V0 = np.array([0.04, 0.05])      # Initial variances (assumed)
mu = np.array([0.02, 0.02])      # Drift terms (risk-free rate)
kappa = np.array([1.0, 1.0])     # Mean reversion speeds
theta = np.array([0.04, 0.05])   # Long-term variances
sigma_v = np.array([0.3, 0.3])   # Vol of vol
rho = np.array([-0.7, -0.7])     # Correlation between asset price and its variance

# Correlation between assets' price processes
rho_assets = np.array([[1.0, 0.5],
                       [0.5, 1.0]])

# Simulation parameters
T = 1.25      # Time to maturity (15 months)
N = 315       # Number of time steps (daily steps over 15 months)
nsims = 1000  # Number of simulations


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


def calculate_payoff(pathS1, pathS2, terminalS1, terminalS2, initinvestment, initpriceS1=549.60, initpriceS2=240.40,
                     barrierS1=329.76, barrierS2=144.24, conversionratioS1=1.8195, conversionratioS2=4.1597):
    '''
    initpriceS1 := Initial price of stock S1 defined to be LONZA GROUP AG (Float)
    initpriceS2 := Initial price of stock S2 defined to be SIKA AG  (Float)
    pathS1 := Stock price path of S1 (list)
    pathS2 := Stock price path of S2 (list)
    terminalS1 := Terminal price of stock S1 on the final fixing date (Float)
    terminalS2 := Terminal price of stock S2 on the final fixing date (Float)
    barrierS1  := Given barrier price of S1 (Float)
    barrierS2 := Given barrier price of S2 (Float)
    '''
    condA = terminalS1 >= initpriceS1 and terminalS2 >= initpriceS2  # condition where terminal price of S1 and S2 are above their initial level on the initial fixing date
    condB = min(pathS1) >= barrierS1 and min(
        pathS2) >= barrierS2  # condition where none of the barriers have been reached
    if condA or condB:
        payoff = (1 + (0.08 / 12 * 15)) * initinvestment + initinvestment
    elif not condA and not condB:
        receiving = min(terminalS1, terminalS2)
        if receiving == terminalS1:
            conversionratio = conversionratioS1
            price = terminalS1
        else:
            conversionratio = conversionratioS2
            price = terminalS2
        payoff = (1 + (0.08 / 12 * 15)) * initinvestment + initinvestment * conversionratio
    elif terminalS1 == 0 or terminalS2 == 0:
        payoff = (1 + (0.08 / 12 * 15)) * initinvestment

    return payoff


def payoff(paths1, paths2, params, fdos):
    """
    Calculates the payoff for each simulation based on the product conditions.

    Params:
        paths1: DataFrame of simulated paths for Lonza (columns: simulations)
        paths2: DataFrame of simulated paths for Sika (columns: simulations)
        params: Dictionary containing product parameters
        fdos: First date of simulation (pricing date)

    Returns:
        payoffs: Array of payoffs for each simulation
    """
    import numpy as np
    from dateutil.relativedelta import relativedelta

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

    payoffs = []

    for sim_name in simulations:
        early_redeem = False
        lonza = paths1[sim_name]
        sika = paths2[sim_name]
        early_observation_dates = dates.get_early_observation_dates(cs.initial_fixing_date, cs.final_fixing_date)

        for date_idx, t_date in enumerate(early_observation_dates):
            if (lonza.loc[t_date] >= cs.initialS1) and (sika.loc[t_date] >= cs.initialS2):
                # Early redemption
                settlement_delay = params.get('Settlement_Delay', 2)
                early_redemption_date = dates.add_business_days(t_date, settlement_delay)
                periods = get_number_of_coupon_periods(fdos, early_redemption_date)
                payoff_value = params['Denomination'] * (1 + params['Coupon_Rate'] * periods)
                payoffs.append(payoff_value)
                early_redeem = True
                break  # Exit the early observation loop

        if not early_redeem:
            barrierhit = checkbarrier(lonza, sika, cs.barrierS1, cs.barrierS2)
            terminallower = check_terminal(lonza, sika, cs.initialS1, cs.initialS2)

            if not barrierhit and not terminallower:  # Best case scenario
                periods = get_number_of_coupon_periods(cs.initial_fixing_date, cs.final_fixing_date)
                payoff_value = params['Denomination'] * (1 + params['Coupon_Rate'] * periods)
                payoffs.append(payoff_value)
            else:
                # Worst-case scenario
                perf_lonza = lonza.iloc[-1] / cs.initialS1
                perf_sika = sika.iloc[-1] / cs.initialS2
                worse_perf = min(perf_lonza, perf_sika)
                periods = get_number_of_coupon_periods(cs.initial_fixing_date, cs.final_fixing_date)
                payoff_value = params['Denomination'] * worse_perf
                payoff_value += params['Denomination'] * params['Coupon_Rate'] * periods
                payoffs.append(payoff_value)

    payoffs = np.array(payoffs)
    return payoffs


# Create date range
start_date = pd.Timestamp('2023-04-30')  # Example start date
dates = pd.bdate_range(start=start_date, periods=N+1)

# Ensure dates match the time grid
if len(dates) != N+1:
    dates = pd.date_range(start=start_date, periods=N+1, freq='D')

# Convert simulated prices to DataFrames
lonza_paths = pd.DataFrame(S_sim[0].T, index=dates)
sika_paths = pd.DataFrame(S_sim[1].T, index=dates)

# Define product parameters
params_product = {
    'Denomination': 1000,            # Denomination of the product
    'Coupon_Rate': (0.08 / 12),      # Monthly coupon rate
    'Settlement_Delay': 2            # Settlement delay in days
}

# First date of simulation
fdos = dates[0]

# Assuming cs and dates modules/functions are properly defined
# For this example, we'll define required constants
class cs:
    initial_fixing_date = dates[0]
    final_fixing_date = dates[-1]
    initialS1 = S0[0]
    initialS2 = S0[1]
    barrierS1 = 0.6 * S0[0]  # 60% barrier level
    barrierS2 = 0.6 * S0[1]
    ticker_list = ['Lonza', 'Sika']
    num_ticker = 2

# Mock dates functions
def get_early_observation_dates(start_date, end_date):
    # For simplicity, assume quarterly observations
    return pd.date_range(start=start_date, end=end_date, freq='3M')

def add_business_days(date, days):
    return pd.bdate_range(start=date, periods=days+1)[-1]

dates.get_early_observation_dates = get_early_observation_dates
dates.add_business_days = add_business_days

# Calculate payoffs
payoffs = payoff(lonza_paths, sika_paths, params_product, fdos)

#Lonza paths
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(lonza_paths.index, lonza_paths.iloc[:, i], label=f'Simulation {i+1}')
plt.title('Simulated Stock Price Paths for Lonza')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

#Sika paths
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(sika_paths.index, sika_paths.iloc[:, i], label=f'Simulation {i+1}')
plt.title('Simulated Stock Price Paths for Sika')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

#payoff distribution
plt.figure(figsize=(10, 6))
plt.hist(payoffs, bins=50, edgecolor='k', alpha=0.7)
plt.title('Distribution of Structured Product Payoffs')
plt.xlabel('Payoff')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Extract terminal prices
lonza_terminal = lonza_paths.iloc[-1, :].values
sika_terminal = sika_paths.iloc[-1, :].values

# Worst performance
worst_perf = np.minimum(lonza_terminal / cs.initialS1, sika_terminal / cs.initialS2)

plt.figure(figsize=(10, 6))
plt.scatter(worst_perf, payoffs, alpha=0.5)
plt.title('Payoff vs. Worst Asset Performance')
plt.xlabel('Worst Asset Performance')
plt.ylabel('Payoff')
plt.grid(True)
plt.show()

# Step 7: Plot stock price paths

# Define the number of simulations to plot
num_samples = 5  # Adjust as needed

# Plot sample stock price paths for Lonza and Sika
plt.figure(figsize=(12, 6))
for i in range(num_samples):
    sim_name = lonza_paths.columns[i]
    plt.plot(lonza_paths.index, lonza_paths[sim_name], label=f'Lonza Sim {i+1}')
    plt.plot(sika_paths.index, sika_paths[sim_name], label=f'Sika Sim {i+1}', linestyle='--')
plt.title('Sample Simulated Stock Price Paths for Lonza and Sika')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


