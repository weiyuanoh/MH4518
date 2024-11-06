import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import constants as cs
import Dates as da
from SimStockPrice import lonza_paths, sika_paths

# Simulation parameters
fdos = cs.initial_fixing_date

# Define product parameters
params_product = {
    'Denomination': 1000,            # Denomination of the product
    'Coupon_Rate': (0.08 / 4),      # Monthly coupon rate
}


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
        early_observation_dates = da.get_early_observation_dates(cs.initial_fixing_date, cs.final_fixing_date)

        for date_idx, t_date in enumerate(early_observation_dates):
            if (lonza.loc[t_date] >= cs.initialS1) and (sika.loc[t_date] >= cs.initialS2):
                # Early redemption
                settlement_delay = params.get('Settlement_Delay', 2)
                early_redemption_date = da.add_business_days(t_date, settlement_delay)
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


def E_payoff(lonza_paths, sika_paths, params_product, fdos):
    expected_payoffs =[]
    individual_payoffs = payoff(lonza_paths, sika_paths, params_product, fdos)
    avg_payoff = np.mean(individual_payoffs)
    expected_payoffs.append(avg_payoff)
    return expected_payoffs

expected_payoff = E_payoff(lonza_paths, sika_paths, params_product, fdos)
print(expected_payoff)


def E_payoff_per_date(lonza_paths, sika_paths, params_product, trading_dates):
    """
    Calculates the expected payoff for each trading date.

    Params:
        lonza_paths: DataFrame of Lonza paths
        sika_paths: DataFrame of Sika paths
        params_product: Dictionary of product parameters
        trading_dates: List of trading dates to calculate expected payoffs for

    Returns:
        expected_payoffs_df: DataFrame mapping each trading date to its expected payoff
    """
    expected_payoffs = []

    for current_date in trading_dates:
        print(f"\nProcessing Trading Date: {current_date.date()}")

        # Slice the paths up to the current trading date
        lonza_subset = lonza_paths.loc[:current_date]
        sika_subset = sika_paths.loc[:current_date]

        # Check if the subsets are non-empty
        if lonza_subset.empty or sika_subset.empty:
            print(f"Empty subset for date: {current_date.date()}. Skipping.")
            expected_payoffs.append({'Trading Date': current_date, 'Expected Payoff': np.nan})
            continue

        # Call the payoff function with the sliced paths
        payoffs = payoff(lonza_subset, sika_subset, params_product, current_date)

        # Debug: Print payoffs
        print(f"Payoffs for {current_date.date()}: {payoffs}")

        # Compute the average payoff for the current trading date
        if len(payoffs) > 0:
            avg_payoff = np.mean(payoffs)
            print(f"Average Payoff for {current_date.date()}: {avg_payoff}")
        else:
            avg_payoff = np.nan  # Handle cases with no payoffs
            print(f"No payoffs returned for {current_date.date()}. Setting average payoff to NaN.")

        # Append the result with the current date
        expected_payoffs.append({'Trading Date': current_date, 'Expected Payoff': avg_payoff})

    # Convert the list of dictionaries to a DataFrame
    expected_payoffs_df = pd.DataFrame(expected_payoffs)
    return expected_payoffs_df

# Retrieve trading dates
trading_dates = da.get_list_dates(cs.initial_fixing_date, cs.final_fixing_date)
print(f"Trading Dates: {trading_dates}")

# Call the new E_payoff_per_date function
expected_payoff_df = E_payoff_per_date(lonza_paths, sika_paths, params_product, trading_dates)

# Print the DataFrame to verify contents
print(expected_payoff_df)

# Optional: Plot the expected payoffs over trading dates
plt.figure(figsize=(12, 6))
plt.plot(expected_payoff_df['Trading Date'], expected_payoff_df['Expected Payoff'], marker='o')
plt.title('Expected Payoff Over Trading Dates')
plt.xlabel('Trading Date')
plt.ylabel('Expected Payoff')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()