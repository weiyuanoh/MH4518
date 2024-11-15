import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yfin
import MH4518.WY.Utils.constants as cs
import MH4518.WY.Utils.Dates as dates 
import MH4518.WY.Utils.payoff as pf
import MH4518.WY.Utils.GBM as gbm
import MH4518.WY.Utils.ProductPrice as pp 
import MH4518.WY.Utils.VarianceRedFunc as vr

import warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp
import logging

# Define RMSE function
import numpy as np
import pandas as pd

def compute_rmse(estimates, realized):
    """
    Compute the Root Mean Square Error (RMSE) between estimates and realized values,
    excluding any pairs with NaN values.
    
    Params:
        estimates (np.ndarray or pd.Series): Estimated values.
        realized (float or np.ndarray or pd.Series): Realized values.
    
    Returns:
        rmse (float): The computed RMSE of the valid data points.
                     Returns np.nan if no valid data points are available.
    """
    # Convert estimates and realized to numpy arrays for consistency
    estimates = np.array(estimates)
    
    if isinstance(realized, pd.Series):
        realized = realized.values
    elif isinstance(realized, (list, np.ndarray)):
        realized = np.array(realized)
    else:
        
        realized = np.full_like(estimates, realized, dtype=np.float64)
    
    # Ensure both arrays have the same length
    if estimates.shape != realized.shape:
        raise ValueError("Estimates and realized values must have the same shape.")
    
    # Create a boolean mask where neither estimates nor realized are NaN
    valid_mask = ~np.isnan(estimates) & ~np.isnan(realized)
    
    # Check if there are any valid data points
    if not np.any(valid_mask):
        print("Warning: No valid data points available for RMSE calculation.")
        return np.nan
    

    valid_estimates = estimates[valid_mask]
    valid_realized = realized[valid_mask]
    
  
    mse = np.mean((valid_estimates - valid_realized) ** 2)

    rmse = np.sqrt(mse)
    
    return rmse


def validate_data(estimates, realized):
    """
    Validates the input data for RMSE computation.
    
    Params:
        estimates (np.ndarray or pd.Series): Estimated values.
        realized (float or np.ndarray or pd.Series): Realized values.
    
    Returns:
        bool: True if data is valid, False otherwise.
    """
    estimates = np.array(estimates)
    
    if isinstance(realized, pd.Series):
        realized = realized.values
    elif isinstance(realized, (list, np.ndarray)):
        realized = np.array(realized)
    else:
        realized = np.full_like(estimates, realized, dtype=np.float64)
    
    # Check for shape compatibility
    if estimates.shape != realized.shape:
        print("Error: Estimates and realized values must have the same shape.")
        return False
    
    if np.isnan(estimates).all():
        print("Error: All estimates are NaN.")
        return False
    if np.isnan(realized).all():
        print("Error: All realized values are NaN.")
        return False
    
    return True

def getdata():
    data = yfin.download(['LONN.SW', 'SIKA.SW'], period='max')['Adj Close']
    data.index = data.index.tz_localize(None)
    data = data.loc[cs.initial_fixing_date: cs.final_fixing_date]
    return data 

data = getdata()
params_product = {
    'Denomination': 1000,
    'Coupon_Rate': (0.08 / 4),  
}

Tlist = dates.num_business_days(cs.initial_fixing_date, cs.final_fixing_date)
present_value_list = []
indexlist = []

def process_fdos(args):
    fdos = args  # Only pass 'fdos'
    try:
        # Set up logging for the child process
        logger = logging.getLogger(f'Process-{fdos}')
        logger.info(f"Processing FDOS: {fdos}")
        cs.n_sims = 1000  # Increased sims for better accuracy
        
        np.random.seed(42 + hash(fdos) % 10000)
        
        sim_T = gbm.multi_asset_gbm_n_sims(
            plot=False,
            plotasset=False,
            nsims=cs.n_sims,
            data=data,
            fdos=fdos
        )
        # Extract asset paths
        lonza_path = sim_T['LONN.SW']
        sika_path = sim_T['SIKA.SW']
        
        # Calculate payoffs using different variance reduction techniques
        # 1. Standard Monte Carlo Payoff
        payoff_mc = pf.payoff(paths1=lonza_path, paths2=sika_path, params=params_product, fdos=fdos)
        
        # 2. Control Variate Payoff
        #payoff_cv = vr.cv(data=data, lonza_path=lonza_path, sika_path=sika_path, fdos=fdos, payoffs_gbm=payoff_mc)
        payoff_cv2 = vr.cv2(payoff_gbm=payoff_mc, data=data, fdos=fdos, original_sika=lonza_path)
        
        # 3. Empirical Martingale Correction Payoff
        payoff_EMC = vr.EMC(fdos=fdos, params_product=params_product, sim_T=sim_T, payoff_original=payoff_mc, data=data)
        
        # Compute expected payoffs
        expected_payoff_mc = np.mean(payoff_mc)
        expected_payoff_cv = np.mean(payoff_cv2)
        expected_payoff_EMC = np.mean(payoff_EMC)
        
        # Discount to present value (assuming present value as of fdos)
        r = cs.interest_rate  
        T_discount = dates.num_business_days(fdos, cs.final_fixing_date) / 252 
        
        present_value_mc = expected_payoff_mc * np.exp(-r * T_discount)
        present_value_cv = expected_payoff_cv * np.exp(-r * T_discount)
        present_value_EMC = expected_payoff_EMC * np.exp(-r * T_discount)
        
        # Return all present values as a tuple
        return present_value_mc, present_value_cv, present_value_EMC
    except Exception as e:
        print(f"Error processing FDOS {fdos}: {e}")
        return None 

def main():
    # Configure logging in the main process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s'
    )
    
    # Generate the list of dates
    date_list = dates.get_list_dates(cs.initial_fixing_date, cs.final_fixing_date)
    date_list = pd.Series(date_list).tolist()
    # date_list = date_list[:120]  
    

    num_processes = mp.cpu_count()
    with mp.Pool(processes=num_processes) as pool:

        # This will return a list of tuples: [(mc1, cv1, EMC1), (mc2, cv2, EMC2), ...]
        results = pool.map(process_fdos, date_list)
    
    # Filter out any None results due to errors
    results = [res for res in results if res is not None]
    
   
    present_value_mc_list, present_value_cv_list, present_value_EMC_list = zip(*results)
    
    # Convert present_value lists to pandas Series with date_list as index
    present_value_mc_series = pd.Series(present_value_mc_list, index=date_list[:len(present_value_mc_list)])
    present_value_cv_series = pd.Series(present_value_cv_list, index=date_list[:len(present_value_cv_list)])
    present_value_EMC_series = pd.Series(present_value_EMC_list, index=date_list[:len(present_value_EMC_list)])
    
    # Obtaining Realized Price 
    realized_price = pp.product_price()  
    
 
    if isinstance(realized_price, pd.Series):
        realized_price_series = realized_price
    else:
        
        realized_price_series = pd.Series([realized_price], index=[date_list[-1]])
    
   
    if isinstance(realized_price_series, pd.Series) and len(realized_price_series) > 1:
        # Find the intersection of dates
        common_dates = present_value_mc_series.index.intersection(realized_price_series.index)
        
        # Align the data based on common dates
        aligned_present_value_mc = present_value_mc_series.loc[common_dates]
        aligned_present_value_cv = present_value_cv_series.loc[common_dates]
        aligned_present_value_EMC = present_value_EMC_series.loc[common_dates]
        aligned_realized_price = realized_price_series.loc[common_dates]
        
        # Compute RMSE between aligned estimates and realized price
        rmse_mc = compute_rmse(aligned_present_value_mc, aligned_realized_price)
        rmse_cv = compute_rmse(aligned_present_value_cv, aligned_realized_price)
        rmse_emc = compute_rmse(aligned_present_value_EMC, aligned_realized_price)
    else:
        # Assume realized_price is a scalar
        rmse_mc = compute_rmse(present_value_mc_list, realized_price)
        rmse_cv = compute_rmse(present_value_cv_list, realized_price)
        rmse_emc = compute_rmse(present_value_EMC_list, realized_price)
    
 
    if isinstance(realized_price_series, pd.Series) and len(realized_price_series) > 1:
        print(f"RMSE (MC vs Realized Price): {rmse_mc:.4f}")
        print(f"RMSE (CV vs Realized Price): {rmse_cv:.4f}")
        print(f"RMSE (EMC vs Realized Price): {rmse_emc:.4f}")
    else:
        print(f"RMSE (MC vs Realized Price): {rmse_mc:.4f}")
        print(f"RMSE (CV vs Realized Price): {rmse_cv:.4f}")
        print(f"RMSE (EMC vs Realized Price): {rmse_emc:.4f}")
    

    plt.figure(figsize=(12, 8))
    
    if isinstance(realized_price_series, pd.Series) and len(realized_price_series) > 1:
        # Plot aligned present values
        plt.plot(aligned_present_value_mc.index, aligned_present_value_mc.values, label='Monte Carlo (MC)', alpha=0.7)
        plt.plot(aligned_present_value_cv.index, aligned_present_value_cv.values, label='Control Variate (CV)', alpha=0.7)
        plt.plot(aligned_present_value_EMC.index, aligned_present_value_EMC.values, label='Empirical Martingale Correction (EMC)', alpha=0.7)
        plt.plot(aligned_realized_price.index, aligned_realized_price.values, label='Realized Price', color='r')
    else:
        # Plot all present values against dates
        plt.plot(date_list[:len(present_value_mc_list)], present_value_mc_list, label='Monte Carlo (MC)', alpha=0.7)
        plt.plot(date_list[:len(present_value_cv_list)], present_value_cv_list, label='Control Variate (CV)', alpha=0.7)
        plt.plot(date_list[:len(present_value_EMC_list)], present_value_EMC_list, label='Empirical Martingale Correction (EMC)', alpha=0.7)
        plt.axhline(y=realized_price, color='r', linestyle='-', label='Realized Price')
    
    plt.xlabel('FDOS')
    plt.ylabel('Present Value')
    plt.title('Present Value Estimates Using Various Variance Reduction Techniques')
    plt.legend()
    plt.grid(True)
    file_path = r"C:\Users\Admin\PycharmProjects\Simulation Techniques in Finance\.venv\MH4518\WY\Graphs\VarianceReduction_nsims1000.png"
    plt.savefig(file_path)
    plt.show()

if __name__ == '__main__':
    main()
    