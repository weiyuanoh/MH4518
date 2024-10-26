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

def checkbarrier(sim: pd.DataFrame, barrierS1: int, barrierS2):
    """
    Takes in 1 simulated path of 2 assets and check if any of the barriers have been reached 


    Returns:
        True if at least one of the barriers has been breached throught the lifetime of this simulaation
    """
    barrierhit = False
    s1_path = sim['LONN.SW']
    s2_path = sim['SIKA.SW']
    if  min(s1_path) < barrierS1 or min(s2_path) < barrierS2:
        barrierhit = True 
    else: 
        barrierhit = False 
    return barrierhit 


def check_terminal(sim: pd.DataFrame, initialS1: float, initialS2: float):
    """
    Takes in 1 simulated path of 2 assets and check if the terminal price < initial price on initial fixing date 

    Returns:
        True if one of the assets has terminal price lower then the initial price 

    """
    terminallower = False
    terminalS1 = sim['LONN.SW'][-1]
    terminalS2 = sim['SIKA.SW'][-1]
    if terminalS1 < initialS1 or terminalS2 < initialS2:
        terminallower = True 
    else:
        terminallower = False

    return terminallower

def autocall(sim, )

def payoff(sim , checkbarrier):
    """
    Takes in 1 simulated path of 2 assets and check if any of the barriers have been reached 

    Returns:
        
    """
    if 

    pass 




def discounting_to_present_1(sims: pd.dataFrame):

    """
    Takes in a dataframe of simulated results from stock price 
    1. checks for barriers if they have been reached
    2. checks for early redemption 
    """
    pass