import pandas as pd 
import numpy as np 
import constants as cs
import Dates as dates 



def calculate_payoff(pathS1, pathS2, terminalS1, terminalS2, initinvestment, initpriceS1 =  549.60, initpriceS2 = 240.40,  barrierS1 = 329.76, barrierS2= 144.24, conversionratioS1 = 1.8195, conversionratioS2 = 4.1597):
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
    condA = terminalS1 >= initpriceS1 and terminalS2 >= initpriceS2 # condition where terminal price of S1 and S2 are above their initial level on the initial fixing date
    condB =  min(pathS1) >= barrierS1 and min(pathS2) >= barrierS2 # condition where none of the barriers have been reached 
    if condA or condB:  
        payoff = (1 + (0.08/12 * 15)) * initinvestment + initinvestment
    elif not condA and not condB: 
        receiving = min(terminalS1, terminalS2)
        if receiving == terminalS1:
            conversionratio = conversionratioS1
            price = terminalS1
        else:
            conversionratio = conversionratioS2
            price = terminalS2
        payoff = (1 + (0.08/12 * 15)) * initinvestment + initinvestment*conversionratio
    elif terminalS1 == 0 or terminalS2 == 0 :
        payoff = (1 + (0.08/12 * 15)) * initinvestment

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

    lonza_path = paths1
    sika_path = paths2
    M = lonza_path.shape[1]  # Number of simulations
    payoffs = np.zeros(M)

    for i in range(M):  # For each simulation
        early_redeem = False
        lonza = lonza_path.iloc[:, i]
        sika = sika_path.iloc[:, i]
        early_observation_dates = dates.get_early_observation_dates(cs.initial_fixing_date, cs.final_fixing_date)
        
        for date_idx, t_date in enumerate(early_observation_dates):
            if (lonza.loc[t_date] >= cs.initialS1) and (sika.loc[t_date] >= cs.initialS2):
                # Early redemption
                # Assume settlement delay (e.g., 2 business days)
                settlement_delay = params.get('Settlement_Delay', 2)
                early_redemption_date = dates.add_business_days(t_date, settlement_delay)
                periods = get_number_of_coupon_periods(fdos, early_redemption_date)
                payoffs[i] = params['Denomination'] * (1 + params['Coupon_Rate'] * periods)
                early_redeem = True
                break  # Exit the early observation loop

        if not early_redeem:
            barrierhit = checkbarrier(lonza, sika, cs.barrierS1, cs.barrierS2)
            terminallower = check_terminal(lonza, sika, cs.initialS1, cs.initialS2)

            if not barrierhit and not terminallower:  # Best case scenario
                periods = get_number_of_coupon_periods(cs.initial_fixing_date, cs.final_fixing_date)
                payoffs[i] = params['Denomination'] * (1 + params['Coupon_Rate'] * periods)
            else:
                # Worst-case scenario
                perf_lonza = lonza.iloc[-1] / cs.initialS1
                perf_sika = sika.iloc[-1] / cs.initialS2
                worse_perf = min(perf_lonza, perf_sika)
                periods = get_number_of_coupon_periods(cs.initial_fixing_date, cs.final_fixing_date)
                payoffs[i] = params['Denomination'] * worse_perf
                payoffs[i] += params['Denomination'] * params['Coupon_Rate'] * periods

    return payoffs


def discounting_to_present_1(sims: pd.DataFrame):

    """
    Takes in a dataframe of simulated results from stock price 
    1. checks for barriers if they have been reached
    2. checks for early redemption 
    """
    pass



## example usage 

