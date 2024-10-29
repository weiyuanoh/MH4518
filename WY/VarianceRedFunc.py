import pandas as pd 
import numpy as np 
import GBM as gbm 
import constants as cs 
import Dates as dates





def av():
    """
    Constructs an antithetic variate estimator of sigma
    
    """


    pass

def european_option_on_minimum(S1_T, S2_T, K, r, T):
    '''
    Params:
        S1_T: price of Lonza at muturity 
        S2_t: price of Sika at maturity 
        K : Strike price to be used in calculation, set to be the inistal price of Lonza 

        r : interest rate 
        T : Time to maturirt
    
    '''
    min_ST = np.minimum(S1_T, S2_T)
    payoff = np.maximum(min_ST - K, 0)
    option_price = np.exp(-r * T) * payoff
    return option_price

def price_option_on_minimum(payoffs_control_variate):
    """
    Params:
        S0_1 : initial price of Lonza for this FDOS 
        S0_2 : initial price of Sika for this FDOS 
        K : Set as initial price of 
    
    """
    E_Y = np.mean(payoffs_control_variate)
    return E_Y



def cv(lonza_path: pd.DataFrame, sika_path : pd.DataFrame, K : float, fdos :pd.Timestamp, payoffs_gbm: pd.DataFrame):
    """
    Variance reduction on n sims on one specified day of FDOS  
    
    Params:
        lonza_path : dataframe of lonzapath with n columns on one specified FDOS 
        sika_path : datafram of sikapath with n columns on one specified FDOS 
        K : Set to be the initial price of Lonza on this specified FDOS 
        fdos: specified fdos 
        payoff_gbm : dataframe of payoffs for this n sims on this FDOS, note that this is not the expected 
                    payoff of this FDOS --> we havent taken the mean

    """
    lonza_terminal_list = lonza_path.iloc[-1].tolist()
    sika_terminal_list = sika_path.iloc[-1].tolist()
    E_Y_list= []
    for i in range(len(lonza_terminal_list)):
        payoffs_control_variate = european_option_on_minimum(S1_T = lonza_terminal_list[i], S2_T=sika_terminal_list[i],
                                                            K = K , r = cs.interest_rate, T = dates.num_business_days(fdos, cs.final_fixing_date))
        E_Y = price_option_on_minimum(payoffs_control_variate)
        E_Y_list.append(E_Y)

    try:
        covariance_matrix = np.cov(payoffs_gbm, E_Y_list)
        cov_XY = covariance_matrix[0, 1]
        var_Y = covariance_matrix[1, 1]
        beta = cov_XY / var_Y

    except Exception as e :
        raise  Exception("shape of payoff_gbm and E_Y_list not equal")
        
    mean_X = np.mean(payoffs_gbm)
    mean_Y = np.mean(E_Y_list)
    theta_CV = mean_X + beta * (E_Y - mean_Y)
    
    var_X = np.var(payoffs_gbm)
    var_theta_CV = var_X - (cov_XY**2) / var_Y
    variance_reduction = (var_X - var_theta_CV) / var_X * 100

    print(f"Variance Reduction Achieved: {variance_reduction:.2f}%")
    return theta_CV

     

def imptsam():
    """
    
    Construst a important sampling estimate of sigma 
    """
    pass 

def emc():
    """
    Does Emprical Martingale Correction 
    
    """
    pass 


