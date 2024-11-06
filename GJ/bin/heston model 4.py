import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import constants as cs
import yfinance as yfin
import Dates as da
import logging
import multiprocessing as mp
import ProductPrice as pp
import SimStockPrice as ssp

# Simulation parameters
fdos = cs.initial_fixing_date

# Define product parameters
params_product = {
    'Denomination': 1000,            # Denomination of the product
    'Coupon_Rate': (0.08 / 4),      # Monthly coupon rate
}


# ===========================
# 4. Define Payoff Functions
# ===========================
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

# ===========================
# 6. Define Expected Price Calculation
# ===========================

def process(fdos):

    # Convert simulated prices to DataFrames
    lonza_paths = pd.DataFrame(ssp.S_sim[0].T)
    sika_paths = pd.DataFrame(ssp.S_sim[1].T)

    #calculate payoff
    payoff_T = payoff(lonza_paths, sika_paths, params_product, fdos)
    #compute expected payoff
    E_payoff = np.mean(payoff_T)
    #discount to PV
    r = cs.interest_rate
    T_discount = da.num_business_days(fdos, cs.final_fixing_date)/252
    PV = E_payoff * np.exp(-r*T_discount)
    return PV


# ===========================
# 7. Execute Simulation and Calculations
# ===========================
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')

    date_list = da.get_list_dates(cs.initial_fixing_date, cs.final_fixing_date)
    date_list = pd.Series(date_list).tolist()

    T = date_list

    # Use a smaller number of simulations for testing
    cs.n_sims = 10  # Adjust as needed

    # Set up the multiprocessing pool
    num_processes = min(mp.cpu_count(), 6)  # Adjust the number of processes
    with mp.Pool(processes=num_processes) as pool:
        # Map the function to the list of dates
        present_value_list = pool.map(process, T)

    # Retrieve product prices if needed elsewhere
    productprice = pp.get_product_price()

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a DataFrame for average payoffs with dates as the index
    present_value_df = pd.DataFrame({'Avg Payoff': present_value_list}, index=T)

    # Plot only the average payoffs
    present_value_df.plot(ax=ax, color='blue', label='Average Payoff')

    # Optional: If you still want to include product prices, do it separately or in a different plot
    # productprice.plot(ax=ax, color='red', label='Product Price')  # Commented out to exclude

    # Customize the plot
    ax.set_title('Average Payoff Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Payoff (Currency Units)')
    ax.legend()
    ax.grid(True)

    # Display the plot
    plt.show()


