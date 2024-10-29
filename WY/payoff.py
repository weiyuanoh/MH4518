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

def checkbarrier(simS1 : pd.DataFrame, simS2 : pd.DataFrame, barrierS1: int, barrierS2):
    """
    Takes in 1 simulated path of 2 assets and check if any of the barriers have been reached 


    Returns:
        True if at least one of the barriers has been breached throught the lifetime of this simulaation
    """

    if  min(simS1) < barrierS1 or min(simS2) < barrierS2:
        barrierhit = True 
    else: 
        barrierhit = False 
    return barrierhit




def check_terminal(simS1 : pd.DataFrame, simS2 : pd.DataFrame, initialS1: float, initialS2: float):
    """
    Takes in seperated dataframe of each assets simulated paths and check if the terminal price of each simulation above the initial level

    Returns:
        True if one of the assets has terminal price lower then the initial price 

    """
    terminallower = False
    terminalS1 = simS1[-1]
    terminalS2 = simS2[-1]
    if terminalS1 < initialS1 or terminalS2 < initialS2:
        terminallower = True 
    else:
        terminallower = False

    return terminallower

def autocall(sim : pd.DataFrame, earlyredemption: list) :   
    """
    Takes in a simulated path and check if early redemption is possible at each early redemption date 

    
    """
    earlyredemption = False 
    for date in earlyredemption:
        vectorprice = sim.loc[date]
        lonza_price = vectorprice[0]
        sika_price = vectorprice[1]
        if lonza_price >= cs.initialS1 and sika_price >= cs.initialS2:
            earlyredemption = True 
    return earlyredemption

def payoff(paths1, paths2 , params, fdos):
    """
    Takes in 1 simulated path of 2 assets and check if any of the barriers have been reached 

    Returns:
        
    """
    
    
        # check for early redemption 
    lonza_path = paths1 
    sika_path = paths2 
    M = lonza_path.shape[1] # number of simulations 
    N = lonza_path.shape[0] - 1 # number of time steps 
    payoffs = np.zeros(M)
    dt = cs.dt
    for i in range(M): # for each simulation 
        early_redeem = False 
        lonza = lonza_path.iloc[:,i]
        sika = sika_path.iloc[:, i]
        for date , t_dates in enumerate(dates.get_early_observation_dates(cs.initial_fixing_date, cs.final_fixing_date)):
            if (lonza.loc[t_dates] >= cs.initialS1 and sika.loc[t_dates] >= cs.initialS2):
                # early redemption
                early_redemption_date = dates.add_business_days(date= t_dates)
                payoffs[i] = params['Denomination']* (1 + params['Coupon_Rate'] * dates.num_business_days(fdos, early_redemption_date))
                early_redeem = True 
                break 
        if not early_redeem :
            barrierhit = checkbarrier(lonza, sika, cs.barrierS1, cs.barrierS2)
            terminallower = check_terminal(lonza, sika, cs.initialS1, cs.initialS2)

            if barrierhit == False and terminallower == False : # best case scenario
                payoffs[i] = params['Denomination'] * (1 + params['Coupon_Rate'] * dates.num_business_days(cs.initial_fixing_date, cs.final_fixing_date))
            else: 
                perf_lonza = lonza[-1]/ cs.initialS1
                perf_sika = sika[-1]/cs.initialS2

                worse_perf = min(perf_lonza, perf_sika)
                payoffs[i] = params['Denomination'] * worse_perf
                payoffs[i] += params['Denomination'] * params['Coupon_Rate'] * dates.num_business_days(cs.initial_fixing_date, cs.final_fixing_date)
    return payoffs


def discounting_to_present_1(sims: pd.DataFrame):

    """
    Takes in a dataframe of simulated results from stock price 
    1. checks for barriers if they have been reached
    2. checks for early redemption 
    """
    pass



## example usage 

