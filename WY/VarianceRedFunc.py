import pandas as pd 
import numpy as np 
import GBM as gbm 
import constants as cs 
import Dates as dates
import yfinance as yfin




def av():
    """
    Constructs an antithetic variate estimator of sigma
    
    """


    pass

def european_option_on_minimum(S1_T, S2_T, K, r, T):
    '''
    Computes the discounted payoff of a European call option on the maximum of two assets.

    Params:
        S1_T: Array of Lonza prices at maturity
        S2_T: Array of Sika prices at maturity
        K: Strike price
        r: Interest rate
        T: Time to maturity in years
    '''
    min_ST = np.minimum(S1_T, S2_T)
    payoff = np.maximum(min_ST - K, 0)
    option_price = np.exp(-r * T) * payoff
    return option_price


import numpy as np
import scipy
from scipy.stats import norm

def analytical_price_option_on_minimum(S0_1, S0_2, K, r, T, sigma1, sigma2, rho, q1=0.77, q2=1.21):
    """
    Computes the analytical price of a European call option on the minimum of two assets.

    Params:
        S0_1: Initial price of asset 1 (S1)
        S0_2: Initial price of asset 2 (S2)
        K: Strike price
        r: Risk-free interest rate
        T: Time to maturity in years
        sigma1: Volatility of asset 1
        sigma2: Volatility of asset 2
        rho: Correlation between asset 1 and asset 2
        q1: Dividend yield of asset 1 (default is 0)
        q2: Dividend yield of asset 2 (default is 0)

    Returns:
        C: Analytical price of the option
    """
    # Compute d1 and d2
    d1 = (np.log(S0_1 / K) + (r - q1 + 0.5 * sigma1 ** 2) * T) / (sigma1 * np.sqrt(T))
    d2 = (np.log(S0_2 / K) + (r - q2 + 0.5 * sigma2 ** 2) * T) / (sigma2 * np.sqrt(T))

    # Adjusted d1 and d2 for bivariate normal CDF
    d1_adj = d1 - sigma1 * np.sqrt(T)
    d2_adj = d2 - sigma2 * np.sqrt(T)

    # Compute the terms
    term1 = S0_1 * np.exp(-q1 * T) * norm.cdf(d1)
    term2 = S0_2 * np.exp(-q2 * T) * norm.cdf(d2)

    # Compute the bivariate normal CDF N2(d1_adj, d2_adj, rho)
    lower = [-np.inf, -np.inf]
    upper = [d1_adj, d2_adj]
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]

    # Compute the bivariate normal CDF
    p, info = scipy.stats._mvn.mvnun(lower, upper, mean, cov)
    N2 = p

    term3 = K * np.exp(-r * T) * N2

    # Compute the option price
    C = term1 + term2 - term3

    return C





def cv(data, lonza_path: pd.DataFrame, sika_path: pd.DataFrame, fdos: pd.Timestamp, payoffs_gbm: np.ndarray):
    # Extract terminal asset prices
    lonza_terminal = lonza_path.iloc[-1].values  # Array of terminal prices
    sika_terminal = sika_path.iloc[-1].values    # Array of terminal prices

    # Time to maturity in years
    T = dates.num_business_days(fdos, cs.final_fixing_date) / 252
    # Define K
    K = data.loc[fdos][0]  * 0.4 # Adjust index if necessary

    # Calculate control variate payoffs for all simulations
    payoffs_control_variate = european_option_on_minimum(
        S1_T=lonza_terminal,
        S2_T=sika_terminal,
        K=K,
        r=cs.interest_rate,
        T=T
    )

    # Ensure payoffs_control_variate is an array
    payoffs_control_variate = np.array(payoffs_control_variate).astype(float)
    print(payoffs_control_variate)

    # Calculate volatilities and correlation
    log_returns = np.log(data / data.shift(1)).dropna()

    # Calculate daily volatilities
    sigma1_daily = log_returns['LONN.SW'].std()
    sigma2_daily = log_returns['SIKA.SW'].std()

    # Annualize the volatilities
    N = 252
    sigma1 = sigma1_daily * np.sqrt(N)
    sigma2 = sigma2_daily * np.sqrt(N)

    # Calculate correlation
    rho = log_returns['LONN.SW'].corr(log_returns['SIKA.SW'])

    # Compute E[Y] analytically
    E_Y = analytical_price_option_on_minimum(
        S0_1=lonza_path.iloc[0, 0],
        S0_2=sika_path.iloc[0, 0],
        K=K,
        r=cs.interest_rate,
        T=T,
        sigma1=sigma1,
        sigma2=sigma2,
        rho=rho
    )

    print(f"E_Y (Analytical Expected Value): {E_Y}")
    

    # Compute sample means
    mean_X = np.mean(payoffs_gbm)
    mean_Y = np.mean(payoffs_control_variate)
    print(f"Mean of Simulated Control Variate Payoffs (bar_Y): {mean_Y}")
    # Compute covariance and variance
    cov_matrix = np.cov(payoffs_gbm, payoffs_control_variate, ddof=1)
    cov_XY = cov_matrix[0, 1]
    var_X = np.var(payoffs_gbm, ddof=1)
    var_Y = cov_matrix[1, 1]
    corr_XY = cov_XY / np.sqrt(var_X * var_Y)
    print(f"Correlation between X and Y: {corr_XY:.4f}")

    # Compute beta
    beta = cov_XY / var_Y
    print(f"Beta (β) Coefficient: {beta}")

    # Compute control variate estimator
    theta_CV = mean_X + beta * (E_Y - mean_Y)
    

    print("NaN in payoffs_gbm:", np.isnan(payoffs_gbm).any())
    print("Inf in payoffs_gbm:", np.isinf(payoffs_gbm).any())
    print("NaN in payoffs_control_variate:", np.isnan(payoffs_control_variate).any())
    print("Inf in payoffs_control_variate:", np.isinf(payoffs_control_variate).any())


    # Variance reduction estimation
    var_X = np.var(payoffs_gbm, ddof=1)
    var_theta_CV = var_X - (cov_XY ** 2) / var_Y
    variance_reduction = (var_X - var_theta_CV) / var_X * 100

    print(f"Variance Reduction Achieved: {variance_reduction:.2f}%")
    print("CV Estimate Payoff :", theta_CV)
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


