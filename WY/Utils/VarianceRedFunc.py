import pandas as pd 
import numpy as np 
import yfinance as yfin
import numpy as np
import scipy
from scipy.stats import norm
import MH4518.WY.Utils.constants as cs
import MH4518.WY.Utils.Dates as dates 
import MH4518.WY.Utils.payoff as pf
import MH4518.WY.Utils.GBM as gbm



def av():
    """
    Constructs an antithetic variate estimator of the p
    
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
    payoff = np.maximum(min_ST-K, 0)
    option_price = np.exp(-r * T) * payoff
    return option_price

def european_option_on_average(S1_T, S2_T, K, r, T):
    '''
    Computes the discounted payoff of a European call option on the average price of two assets.

    Params:
        S1_T: Array of Lonza prices at maturity
        S2_T: Array of Sika prices at maturity
        K: Strike price
        r: Interest rate
        T: Time to maturity in years

    Returns:
        option_price: Array of discounted payoffs for each simulation
    '''
    avg_ST = 0.5 * (S1_T + S2_T)
    payoff = np.maximum(avg_ST - K, 0)
    option_price = np.exp(-r * T) * payoff
    return option_price


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



def european_option_on_average_price_analytical(S0_1, S0_2, K, r, T, sigma_1, sigma_2, rho, q=0.0):
    '''
    Approximates the analytical price of a European call option on the average price of two assets.

    Params:
        S0_1: Initial price of Lonza
        S0_2: Initial price of Sika
        K: Strike price
        r: Risk-free interest rate
        T: Time to maturity
        sigma_1: Volatility of Lonza
        sigma_2: Volatility of Sika
        rho: Correlation between Lonza and Sika
        q: Dividend yield (assumed zero)

    Returns:
        C_avg: Approximate analytical price of the option
    '''
    S_avg = 0.5 * (S0_1 + S0_2)
    sigma_avg = 0.5 * np.sqrt(sigma_1**2 + sigma_2**2 + 2 * rho * sigma_1 * sigma_2)
    d1 = (np.log(S_avg / K) + (r - q + 0.5 * sigma_avg**2) * T) / (sigma_avg * np.sqrt(T))
    d2 = d1 - sigma_avg * np.sqrt(T)
    C_avg = S_avg * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return C_avg






def cv(data, lonza_path: pd.DataFrame, sika_path: pd.DataFrame, fdos: pd.Timestamp, payoffs_gbm: np.ndarray):
    # Extract terminal asset prices
    lonza_terminal = lonza_path.iloc[-1].values  # Array of terminal prices
    sika_terminal = sika_path.iloc[-1].values    # Array of terminal prices

    # Time to maturity in years
    T = dates.num_business_days(fdos, cs.final_fixing_date) / 252
    # Define K
    K = 250 # Adjust index if necessary

    # Calculate control variate payoffs for all simulations
    payoffs_control_variate = european_option_on_average(
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
    S0_1 = lonza_path.iloc[0, 0]  # Assuming first simulation's initial price
    S0_2 = sika_path.iloc[0, 0]   # Assuming first simulation's initial price

    # Compute E[Y] analytically
    E_Y = european_option_on_average_price_analytical(S0_1=S0_1, S0_2=S0_2,
                                                      K = K, 
                                                      r = cs.interest_rate,
                                                      T = T, 
                                                      sigma_1= sigma1,
                                                      sigma_2= sigma2,
                                                      rho = rho,
                                                      q = 0.0113
    )

    print(f"E_Y (Analytical Expected Value): {E_Y}")
    

    # Compute sample means
    mean_X = np.mean(payoffs_gbm)
    mean_Y = np.mean(payoffs_control_variate)

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


    # Variance reduction estimation
    var_X = np.var(payoffs_gbm, ddof=1)
    var_theta_CV = var_X - (cov_XY ** 2) / var_Y
    variance_reduction = (var_X - var_theta_CV) / var_X * 100

    print(f"Variance Reduction Achieved: {variance_reduction:.2f}%")
    print("CV Estimate Payoff :", theta_CV)
    return theta_CV


def cv2(payoff_gbm, data: pd.DataFrame, fdos, original_sika):
    '''
    Takes in simulated payoff_gbm.
    E_Y is the mean of Sika's terminal values from original simulations.
    
    Params:
        payoff_gbm: array-like, payoffs from initial simulations
        data: DataFrame containing initial asset prices
        fdos: First date of simulation (pricing date)
        original_sika: DataFrame of simulated Sika paths
    '''
    params_product = {
        'Denomination': 1000,
        'Coupon_Rate': (0.08 / 4),  # Quarterly coupon payments
    }

    # Compute terminal values
    terminal_original_sika = original_sika.iloc[-1].values
    var_X = np.var(payoff_gbm, ddof=1)
    var_Y = np.var(terminal_original_sika, ddof=1)
    print('Var_X:', var_X)
    print('Var_Y:', var_Y)

    # Check for zero variance
    if var_Y == 0:
        print("Error: Variance of Y (Var_Y) is zero. Cannot compute beta.")
        return np.nan

    # Compute covariance and correlation
    cov_matrix = np.cov(payoff_gbm, terminal_original_sika, ddof=1)
    if cov_matrix.shape != (2, 2):
        print("Error: Covariance matrix shape is incorrect.")
        return np.nan

    cov_XY = cov_matrix[0, 1]
    print('Cov_XY:', cov_XY)

    if np.isnan(cov_XY):
        print("Error: Covariance between X and Y is NaN.")
        return np.nan

    if var_X == 0:
        print("Error: Variance of X (Var_X) is zero. Cannot compute correlation.")
        return np.nan

    corr_XY = cov_XY / np.sqrt(var_X * var_Y)
    print(f"Correlation between X and Y: {corr_XY:.4f}")

    beta = cov_XY / var_Y
    print(f"Beta (β) Coefficient: {beta}")

    # Generate new simulations for Y
    sim_extra = gbm.multi_asset_gbm_n_sims(
        plot=False, 
        plotasset=False, 
        nsims=cs.n_sims, 
        data=data, 
        fdos=fdos
    )  # new set of simulations

    # Extract new Sika and Lonza paths
    sika_path_new = sim_extra['SIKA.SW']
    sika_path_new_terminal = sika_path_new.iloc[-1].values
    lonza_path_new = sim_extra['LONN.SW']
    lonza_path_new_terminal = lonza_path_new.iloc[-1].values

    # Compute payoffs for new simulations
    payoff_extra = pf.payoff(lonza_path_new, sika_path_new, params_product, fdos)  # new set of X
    print('Payoff array:', payoff_extra)

    # Check for NaNs in new payoffs
    if np.isnan(payoff_extra).any():
        print("Error: NaNs found in payoff_extra.")
        return np.nan

    # Compute means
    mean_X = np.mean(payoff_extra)
    mean_Y = np.mean(lonza_path_new_terminal)
    E_Y = np.mean(terminal_original_sika)
    print(f"mean_X: {mean_X}, mean_Y: {mean_Y}, E_Y: {E_Y}")

    # Check for NaNs in means
    if np.isnan(mean_X) or np.isnan(mean_Y) or np.isnan(E_Y):
        print("Error: One of the means is NaN.")
        return np.nan

    # Compute theta_CV
    theta_CV = mean_X + beta * (mean_Y - E_Y)
    print("Correction (mean_Y - E_Y):", mean_Y - E_Y)
    print("CV Estimate Payoff (theta_CV):", theta_CV)

    # Compute variance of theta_CV
    var_theta_CV = var_X - (cov_XY ** 2) / var_Y
    if np.isnan(var_theta_CV):
        print("Error: var_theta_CV is NaN.")
        return np.nan

    # Compute variance reduction
    if var_X == 0:
        print("Error: Var_X is zero. Cannot compute variance reduction.")
        variance_reduction = 0
    else:
        variance_reduction = (var_X - var_theta_CV) / var_X * 100
    print(f"Variance Reduction Achieved: {variance_reduction:.2f}%")

    return theta_CV



def restructure_simulated_paths(sim_T):
    """
    Restructure the MultiIndex DataFrame returned by multi_asset_gbm_n_sims
    to a DataFrame with simulations as rows and assets as columns.

    Params:
        sim_T (pd.DataFrame): MultiIndex DataFrame with asset and simulation levels.

    Returns:
        terminal_prices (pd.DataFrame): DataFrame with simulations as index and assets as columns.
    """
    # Select the terminal row (last date)
    terminal_row = sim_T.iloc[-1]

    # Ensure columns are a MultiIndex
    if not isinstance(terminal_row.index, pd.MultiIndex):
        raise ValueError("Columns of sim_T must be a MultiIndex with levels [Asset, Simulation].")

    # Convert the Series with MultiIndex to DataFrame
    terminal_df = terminal_row.reset_index()
    terminal_df.columns = ['Asset', 'Simulation', 'Price']

    # Pivot the DataFrame to have Simulation as rows and Asset as columns
    terminal_prices = terminal_df.pivot(index='Simulation', columns='Asset', values='Price')

    return terminal_prices

def empirical_martingale_correction(simulated_paths, r, T, S0):
    """
    Applies Empirical Martingale Correction to the simulated asset paths by computing
    a single scaling factor based on individual asset scaling factors.

    Params:
        simulated_paths (pd.DataFrame): Simulated asset paths at maturity for all simulations.
                                        Columns are asset names, rows are simulations.
        r (float): Risk-free interest rate.
        T (float): Time to maturity in years.
        S0 (dict): Initial asset prices, e.g., {'Lonza': 549.60, 'Sika': 240.40}.

    Returns:
        unified_scaling_factor (float): Aggregated scaling factor for all assets.
    """
    discount_factor = np.exp(-r * T)
    scaling_factors = []

    for asset in simulated_paths.columns:
        # Compute discounted simulated prices
        discounted_S_T = discount_factor * simulated_paths[asset].values
        # Empirical mean of discounted prices
        empirical_mean = np.mean(discounted_S_T)
        # Theoretical expectation
        theoretical_mean = S0[asset]
        
        if empirical_mean == 0:
            raise ValueError(f"Empirical mean for asset {asset} is zero. Cannot compute scaling factor.")
        
        # Scaling factor to adjust the empirical mean to match theoretical mean
        scaling_factor = theoretical_mean / empirical_mean
        scaling_factors.append(scaling_factor)

    # Aggregate scaling factors (e.g., arithmetic mean)
    unified_scaling_factor = np.mean(scaling_factors)
    
    return unified_scaling_factor

def adjust_payoffs(payoffs, unified_scaling_factor):
    """
    Adjusts the original payoffs using the unified martingale correction scaling factor.

    Params:
        payoffs (np.ndarray): Original array of payoffs from simulations.
        unified_scaling_factor (float): Unified scaling factor for all assets.

    Returns:
        adjusted_payoffs (np.ndarray): Adjusted array of payoffs.
    """
    # Apply the unified scaling factor multiplicatively
    adjusted_payoffs = payoffs * unified_scaling_factor
    return adjusted_payoffs

def EMC(fdos, params_product, sim_T, payoff_original, data):
    """
    Processes a single FDOS: simulates paths, computes payoffs, applies EMC, and discounts to present value.

    Params:
        fdos: First date of simulation
        params_product: Parameters of the product
        sim_T: Simulated asset paths (MultiIndex DataFrame)
        payoff_original: Original payoffs from simulations (np.ndarray)
        data: DataFrame containing initial asset prices indexed by FDOS

    Returns:
        present_value: Discounted present value of adjusted payoffs
    """
    try:
        # Restructure the simulated paths to get terminal prices
        terminal_prices = restructure_simulated_paths(sim_T)
        
        # Retrieve initial asset prices as a dictionary
        S0 = data.loc[fdos].to_dict()  # e.g., {'Lonza': 549.60, 'Sika': 240.40}
        
        # Calculate time to maturity in years
        T_discount = dates.num_business_days(fdos, cs.final_fixing_date) / 252  # Assuming 252 trading days
        T = T_discount
        
        # Compute the unified scaling factor using multiplicative correction
        unified_scaling_factor = empirical_martingale_correction(
            simulated_paths=terminal_prices,
            r=cs.interest_rate,
            T=T,
            S0=S0
        )
        
        # Adjust payoffs uniformly using the scaling factor
        adjusted_payoffs = adjust_payoffs(payoff_original, unified_scaling_factor)
        
        # Discount adjusted payoffs to present value
        discount_factor = np.exp(-cs.interest_rate * T)
        present_value = discount_factor * np.mean(adjusted_payoffs)
        
        return present_value
    except Exception as e:
        print(f"Error processing FDOS {fdos}: {e}")
        return np.nan  # Return NaN or handle as appropriate



