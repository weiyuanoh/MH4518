import pandas as pd
import matplotlib.pyplot as plt
import logging
import Dates as dates
import numpy as np
from dateutil.relativedelta import relativedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants (replace these with actual values)
class Constants:
    initial_fixing_date = "2023-04-27"
    final_fixing_date = pd.to_datetime("2024-07-30")
    initialS1 = 524.60  # Initial price of Asset 1 (e.g., Lonza)
    initialS2 = 271.60  # Initial price of Asset 2 (e.g., Sika)
    barrierS1 = 329.76
    barrierS2 = 144.24
    V0_S1 = 0.01        # Initial variance of Asset 1
    V0_S2 = 0.195311        # Initial variance of Asset 2
    mu_S1 = 0.018        # Expected return of Asset 1
    mu_S2 = 0.018        # Expected return of Asset 2
    kappa_S1 = 4.483091      # Mean reversion rate of Asset 1
    kappa_S2 = 0.901516      # Mean reversion rate of Asset 2
    theta_S1 = 0.124813     # Long-term variance of Asset 1
    theta_S2 = 0.184639     # Long-term variance of Asset 2
    sigma_v_S1 = 1    # Volatility of variance of Asset 1
    sigma_v_S2 = 0.549895   # Volatility of variance of Asset 2
    rho_S1 = 0.772476       # Correlation between asset price and variance of Asset 1
    rho_S2 = -0.728402       # Correlation between asset price and variance of Asset 2
    rho_assets = np.array([[1.0, 0.4250732997428952], [0.4250732997428952, 1.0]])  # Correlation between assets

cs = Constants()

# Load actual price data for Lonza (Asset 1)
lonza_price_df = pd.read_csv('/Users/guanjuwang/Desktop/TAN /MH4518/lonza_price_90.csv')
# Load actual price data for Sika (Asset 2)
sika_price_df = pd.read_csv('/Users/guanjuwang/Desktop/TAN /MH4518/sika_price_90.csv')
# For Lonza
lonza_price_df['Date'] = pd.to_datetime(lonza_price_df['Date'],format = '%Y-%m-%d')
lonza_price_df.set_index('Date', inplace=True)
lonza_price_df.rename(columns={'Close': 'S1'}, inplace=True)
# For Sika
sika_price_df['Date'] = pd.to_datetime(sika_price_df['Date'],format = '%Y-%m-%d')
sika_price_df.set_index('Date', inplace=True)
sika_price_df.rename(columns={'Close': 'S2'}, inplace=True)
# Keep only the 'S1' column for Lonza and 'S2' column for Sika
lonza_price_df = lonza_price_df[['S1']]
sika_price_df = sika_price_df[['S2']]
# Combine the two DataFrames on the Date index
actual_prices_df = lonza_price_df.join(sika_price_df, how='inner')
# Combine the two DataFrames on the Date index
actual_prices_df = lonza_price_df.join(sika_price_df, how='inner')
last_90_days = lonza_price_df.index[-90:]
# Convert each element to a date object (list of dates)
last_90_days = pd.to_datetime(last_90_days)


num_simulations = 1000
num_days = len(last_90_days)
num_assets = 2  # Two assets


# Initialize matrices to store the simulation results
simulated_prices_S1 = np.zeros((num_simulations, num_days))
simulated_prices_S2 = np.zeros((num_simulations, num_days))

# Heston model parameters
S0_original = [cs.initialS1, cs.initialS2]  # Initial prices
V0_original = [cs.V0_S1, cs.V0_S2]          # Initial variances
mu = [cs.mu_S1, cs.mu_S2]                   # Expected returns
kappa = [cs.kappa_S1, cs.kappa_S2]          # Mean reversion rates
theta = [cs.theta_S1, cs.theta_S2]          # Long-term variance levels
sigma_v = [cs.sigma_v_S1, cs.sigma_v_S2]    # Volatility of variance
rho = [cs.rho_S1, cs.rho_S2]                # Correlation between asset price and variance
rho_assets = cs.rho_assets                  # Correlation matrix between the two assets

dt = 1 / 252  # Daily time step

def simulate_heston_multidim(S0, V0, mu, kappa, theta, sigma_v, rho, rho_assets, N, dt, nsims):
    num_assets = len(S0)
    # Initialize arrays
    S = np.zeros((nsims, N, num_assets))
    V = np.zeros((nsims, N, num_assets))

    # Set initial values
    S[:, 0, :] = S0
    V[:, 0, :] = V0

    # Cholesky decomposition for price and volatilities
    corr_matrix = np.block([
        [rho_assets, np.zeros((num_assets, num_assets))],
        [np.zeros((num_assets, num_assets)), np.eye(num_assets)]
    ])
    L = np.linalg.cholesky(corr_matrix)

    for t in range(1, N):
        Z = np.random.normal(size=(2 * num_assets, nsims))
        dW = L @ Z * np.sqrt(dt)  # Shape: (2 * num_assets, nsims)

        for i in range(num_assets):
            dW_S_i = dW[i, :]
            dW_v_i = dW[num_assets + i, :]
            dW_v_i = rho[i] * dW_S_i + np.sqrt(1 - rho[i] ** 2) * dW_v_i

            V_prev = V[:, t - 1, i]
            V_sqrt = np.sqrt(np.maximum(V_prev, 0))
            V[:, t, i] = V_prev + kappa[i] * (theta[i] - V_prev) * dt + sigma_v[i] * V_sqrt * dW_v_i
            V[:, t, i] = np.maximum(V[:, t, i], 0)

            S_prev = S[:, t - 1, i]
            S[:, t, i] = S_prev * np.exp((mu[i] - 0.5 * V_prev) * dt + V_sqrt * dW_S_i)

    return S, V

# Initialize a list to store full price paths for the first 5 days
full_price_paths_S1 = []
full_price_paths_S2 = []
full_dates = []

# Loop over each trading day
for day_idx, current_date in enumerate(last_90_days):
    logging.info(f'Simulating from day {day_idx + 1}/{num_days} ({current_date.date()})')

    # Remaining days from current date
    remaining_days = last_90_days[last_90_days >= current_date]
    N = len(remaining_days)

    # Use initial prices (you can update S0 if you have observed prices on current_date)
    try:
        # Extract actual prices for current_date
        S0 = [
            actual_prices_df.loc[current_date, 'S1'],
            actual_prices_df.loc[current_date, 'S2']
        ]
    except KeyError:
        # Handle the case where the current_date is not in actual_prices_df
        logging.warning(f"No actual prices found for {current_date.date()}, using original S0.")
        S0 = S0_original
    V0 = V0_original  # If you have updated V0 for each day, update it here

    # Skip if no remaining days
    if N <= 0:
        continue

    # Simulate paths from the current day
    S_simulated, _ = simulate_heston_multidim(S0, V0, mu, kappa, theta, sigma_v, rho, rho_assets, N, dt, num_simulations)

    # Store the first simulated prices (for the current day)
    simulated_prices_S1[:, day_idx] = S_simulated[:, 0, 0]
    simulated_prices_S2[:, day_idx] = S_simulated[:, 0, 1]

    # For the first 5 days, store the full price paths
    if day_idx < 92:
        full_price_paths_S1.append(S_simulated[:, :, 0])
        full_price_paths_S2.append(S_simulated[:, :, 1])
        full_dates.append(remaining_days)

#payoff function
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
        early_observation_dates = dates.get_early_observation_dates(pd.Timestamp('2024-05-07'), cs.final_fixing_date)

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
                periods = get_number_of_coupon_periods(pd.Timestamp('2024-05-07'), cs.final_fixing_date)
                payoff_value = params['Denomination'] * (1 + params['Coupon_Rate'] * periods)
                payoffs.append(payoff_value)
            else:
                # Worst-case scenario
                perf_lonza = lonza.iloc[-1] / cs.initialS1
                perf_sika = sika.iloc[-1] / cs.initialS2
                worse_perf = min(perf_lonza, perf_sika)
                periods = get_number_of_coupon_periods(pd.Timestamp('2024-05-07'), cs.final_fixing_date)
                payoff_value = params['Denomination'] * worse_perf
                payoff_value += params['Denomination'] * params['Coupon_Rate'] * periods
                payoffs.append(payoff_value)

    payoffs = np.array(payoffs)
    return payoffs

# Initialize variables
payoffs_list = []
params = {'Denomination': 1000, 'Coupon_Rate': 0.02}
fdos = last_90_days[0]
batch_size = 1000  # Number of simulations per day

# Loop over each day
for day_idx in range(len(full_price_paths_S1)):
    batch_start_idx = day_idx * batch_size
    sim_indices = range(batch_start_idx, batch_start_idx + batch_size)

    # Get the simulated paths for S1 and S2
    paths_S1_array = full_price_paths_S1[day_idx]
    paths_S2_array = full_price_paths_S2[day_idx]

    # Transpose the arrays to create DataFrames
    paths_S1_df = pd.DataFrame(paths_S1_array.T)
    paths_S2_df = pd.DataFrame(paths_S2_array.T)

    # Set the index to the dates for the current simulation
    dates_for_simulation = full_dates[day_idx]
    paths_S1_df.index = dates_for_simulation
    paths_S2_df.index = dates_for_simulation

    # Assign simulation indices to the columns
    paths_S1_df.columns = sim_indices
    paths_S2_df.columns = sim_indices

    # Call the payoff function
    payoffs = payoff(paths_S1_df, paths_S2_df, params, last_90_days[day_idx])
    # Store the payoffs
    payoffs_list.append(payoffs)

df_payoff = pd.DataFrame(payoffs_list)

# Assuming you have all 90 rows, here we just demonstrate with first 5
r = 0.018

# Create a 'Day' column representing the number of days since the first day
df_payoff['Day'] = df_payoff.index  # Assuming row 0 is Day 0

# Convert days to years
df_payoff['Time_in_Years'] = df_payoff['Day'] / 365

# Calculate the discount factor using continuous discounting
df_payoff['Discount_Factor'] = np.exp(-r * df_payoff['Time_in_Years'])

# Specify the columns to discount
columns_to_discount = list(range(1000))
# Create a copy of the DataFrame to store discounted values
df_discounted = df_payoff.copy()
# Apply the discount factor
df_discounted[columns_to_discount] = df_payoff[columns_to_discount].multiply(df_payoff['Discount_Factor'], axis=0)

df_discounted = df_discounted.drop(columns=['Day', 'Time_in_Years', 'Discount_Factor'])

df_payoff['avg_payoff'] = df_discounted.mean(axis=1)
avg_payoff = df_payoff['avg_payoff']

# Historical Data provided
data = [
    {"date": "2024-07-30", "value": 990.9},
    {"date": "2024-07-29", "value": 991.1},
    {"date": "2024-07-26", "value": 991.5},
    {"date": "2024-07-25", "value": 985.6},
    {"date": "2024-07-24", "value": 940.1},
    {"date": "2024-07-23", "value": 937.8},
    {"date": "2024-07-22", "value": 924.6},
    {"date": "2024-07-19", "value": 915.3},
    {"date": "2024-07-18", "value": 931.4},
    {"date": "2024-07-17", "value": 918.8},
    {"date": "2024-07-16", "value": 927.5},
    {"date": "2024-07-15", "value": 930.4},
    {"date": "2024-07-12", "value": 933.5},
    {"date": "2024-07-11", "value": 937.2},
    {"date": "2024-07-10", "value": 920.5},
    {"date": "2024-07-09", "value": 912.2},
    {"date": "2024-07-08", "value": 908.1},
    {"date": "2024-07-05", "value": 907.7},
    {"date": "2024-07-04", "value": 901.7},
    {"date": "2024-07-03", "value": 898.6},
    {"date": "2024-07-02", "value": 879.9},
    {"date": "2024-07-01", "value": 884.1},
    {"date": "2024-06-28", "value": 887.5},
    {"date": "2024-06-27", "value": 881.2},
    {"date": "2024-06-26", "value": 880.8},
    {"date": "2024-06-25", "value": 876.5},
    {"date": "2024-06-24", "value": 882},
    {"date": "2024-06-21", "value": 865.5},
    {"date": "2024-06-20", "value": 871.1},
    {"date": "2024-06-19", "value": 858.6},
    {"date": "2024-06-18", "value": 864.7},
    {"date": "2024-06-17", "value": 857.3},
    {"date": "2024-06-14", "value": 870.5},
    {"date": "2024-06-13", "value": 872.9},
    {"date": "2024-06-12", "value": 879.2},
    {"date": "2024-06-11", "value": 896},
    {"date": "2024-06-06", "value": 911.3},
    {"date": "2024-06-05", "value": 896},
    {"date": "2024-06-04", "value": 880.2},
    {"date": "2024-06-03", "value": 882.7},
    {"date": "2024-05-31", "value": 882.9},
    {"date": "2024-05-30", "value": 884.6},
    {"date": "2024-05-29", "value": 883.8},
    {"date": "2024-05-28", "value": 918.2},
    {"date": "2024-05-27", "value": 923},
    {"date": "2024-05-24", "value": 925.8},
    {"date": "2024-05-23", "value": 918.3},
    {"date": "2024-05-22", "value": 919.2},
    {"date": "2024-05-21", "value": 942.2},
    {"date": "2024-05-17", "value": 942.4},
    {"date": "2024-05-16", "value": 941.5},
    {"date": "2024-05-15", "value": 929},
    {"date": "2024-05-14", "value": 904.4},
    {"date": "2024-05-13", "value": 922.5},
    {"date": "2024-05-10", "value": 930.8},
    {"date": "2024-05-08", "value": 929.8},
    {"date": "2024-05-07", "value": 921.7},
    {"date": "2024-05-06", "value": 921.6},
    {"date": "2024-05-03", "value": 921.1},
    {"date": "2024-05-02", "value": 919.5},
    {"date": "2024-04-30", "value": 909.6}
]

# Create DataFrame
df = pd.DataFrame(data)
df.sort_values(by='date', inplace=True)
df['date'] = pd.to_datetime(df['date'])

# Combined plot of realized and simulated payoffs
plt.figure(figsize=(12, 6))
# Plotting the realized payoff
plt.plot(df['date'], df['value'], linestyle='-', color='blue', label="Realized Payoff")
# Plotting the simulated payoff
plt.plot(lonza_price_df.index[27:91], avg_payoff[27:91], linestyle='-', color='orange', label="Simulated Payoff")
# Adding title and labels
plt.title("Realized and Simulated Payoffs Over Time")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
# Display grid and format
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


