import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yfin
import MH4518.WY.Utils.constants as cs
import MH4518.WY.Utils.Dates as dates 
import MH4518.WY.Utils.payoff as pf
import MH4518.WY.Utils.GBM as gbm
import MH4518.WY.Utils.ProductPrice as pp 

import warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp
import pandas as pd
import logging

# Ensure all necessary modules and functions are imported or defined
# import gbm, pf, cs, dates, data, params_product

def getdata():
    data = yfin.download(['LONN.SW', 'SIKA.SW'], period= 'max')['Adj Close']
    data.index = data.index.tz_localize(None)
    data = data.loc[cs.initial_fixing_date: cs.final_fixing_date]
    return data 

data = getdata()
params_product = {
    'Denomination': 1000,
    'Coupon_Rate': (0.08 / 4) ,  # Quarterly coupon payments
}

Tlist = dates.num_business_days(cs.initial_fixing_date, cs.final_fixing_date)
present_value_list = []
indexlist = []
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
        # Assume realized is a scalar, create an array filled with the scalar value
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
    
    # Apply the mask to filter out NaN values
    valid_estimates = estimates[valid_mask]
    valid_realized = realized[valid_mask]
    
    # Compute Mean Squared Error (MSE)
    mse = np.mean((valid_estimates - valid_realized) ** 2)
    
    # Compute Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    return rmse

def process_fdos(args):
    fdos = args  # Only pass 'fdos'
    try:
        # Set up logging for the child process
        logger = logging.getLogger(f'Process-{fdos}')
        logger.info(f"Processing FDOS: {fdos}")

        # Load or access necessary data
        # If 'data', 'params_product', 'cs', and 'dates' are global, you can access them directly
        # Alternatively, pass them as arguments or load them within the function

        # Run the simulation
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
        # Calculate payoffs
        payoff_T = pf.payoff(lonza_path, sika_path, params_product, fdos)
        # Compute expected payoff
        expected_payoff = np.mean(payoff_T)
        # Discount to present value (assuming present value as of fdos)
        r = cs.interest_rate
        T_discount = dates.num_business_days(fdos, cs.final_fixing_date) / 252
        present_value = expected_payoff * np.exp(-r * T_discount)
        return present_value
    except Exception as e:
        logger.error(f"Error processing FDOS {fdos}: {e}")
        return None

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')

    date_list = dates.get_list_dates(cs.initial_fixing_date, cs.final_fixing_date)
    date_list = pd.Series(date_list).tolist()
    # For testing, limit the number of dates
    # date_list = date_list[:5]  # Remove or adjust this line in production

    T = date_list

    # Use a smaller number of simulations for testing
    cs.n_sims = 1000 # Adjust as needed

    # Set up the multiprocessing pool
    num_processes = min(mp.cpu_count(), 6)  # Adjust the number of processes
    with mp.Pool(processes=num_processes) as pool:
        # Map the function to the list of dates
        present_value_list = pool.map(process_fdos, T)

    # Optionally, process the results
    print("Present Value List:")
    print(present_value_list)

    productprice = pp.product_price()
    present_value_df = pd.Series(present_value_list, index=T)
    print(present_value_df)


    # Export to CSV 
    file = r'C:\Users\Admin\PycharmProjects\Simulation Techniques in Finance\.venv\MH4518\WY\Graphs\BlackScholesMC_100.csv'
    present_value_df.to_csv(file) 


    if isinstance(productprice, pd.Series) and len(productprice) > 1:
        # Allign dates between Payoff Array and Product Price 
        common_dates = present_value_df.index.intersection(productprice.index)
        present_value_df = present_value_df.loc[common_dates]
        print(present_value_df)
        productprice = productprice.loc[common_dates]

    plt.plot(present_value_df.index, present_value_df.values, label = 'BSM MC')
    plt.plot(productprice.index, productprice.values, label = 'Product Price')
    plt.xlabel('FDOS')
    plt.ylabel('Present Value')
    plt.legend()
    plt.show()

    # Calculate the RMSE between BSM and the Actual Price path

    rmse_mc = compute_rmse(present_value_df, productprice)
    print(f"RMSE (MC vs Realized Price): {rmse_mc:.4f}")

    
 

    