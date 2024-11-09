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
import MH4518.WY.Utils.VarianceRedFunc as vr

import warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp
import pandas as pd
import logging

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

import logging
import numpy as np

def process_fdos(args):
    fdos = args  # Only pass 'fdos'
    try:
        # Set up logging for the child process
        logger = logging.getLogger(f'Process-{fdos}')
        logger.info(f"Processing FDOS: {fdos}")
        cs.n_sims = 1000
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
        
        # Calculate payoffs using different variance reduction techniques
        # 1. Standard Monte Carlo Payoff
        payoff_mc = pf.payoff(paths1=lonza_path, paths2=sika_path, params=params_product, fdos=fdos)
        
        # 2. Control Variate Payoff
        #payoff_cv = vr.cv(data=data, lonza_path=lonza_path, sika_path=sika_path, fdos=fdos, payoffs_gbm=payoff_mc)
        payoff_cv2 = vr.cv2(payoff_gbm= payoff_mc, data = data, fdos= fdos, original_sika=lonza_path)
        
        # 4. Empirical Martingale Correction Payoff
        # Assuming vr.emc is your Empirical Martingale Correction function
        payoff_EMC = vr.EMC(fdos = fdos, params_product= params_product, sim_T=sim_T, payoff_original=payoff_mc)
        
        # Compute expected payoffs
        expected_payoff_mc = np.mean(payoff_mc)
        expected_payoff_cv = np.mean(payoff_cv2)
        expected_payoff_EMC = np.mean(payoff_EMC)
        
        # Discount to present value (assuming present value as of fdos)
        r = cs.interest_rate  # Use the same interest rate as in the simulation
        T_discount = dates.num_business_days(fdos, cs.final_fixing_date) / 252  # Time in years
        
        present_value_mc = expected_payoff_mc * np.exp(-r * T_discount)
        present_value_cv = expected_payoff_cv * np.exp(-r * T_discount)
        present_value_EMC = expected_payoff_EMC * np.exp(-r * T_discount)
        
        # Return all present values as a tuple
        return present_value_mc, present_value_cv, present_value_EMC
    except Exception as e:
        print(f"Error processing FDOS {fdos}: {e}")
        return None  # or some default value

def main():
    # Configure logging in the main process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s'
    )
    
    # Generate the list of dates
    date_list = dates.get_list_dates(cs.initial_fixing_date, cs.final_fixing_date)
    date_list = pd.Series(date_list).tolist()
    # Uncomment the following line to limit the number of dates for testing
    # date_list = date_list[:120]  
    
    # Initialize the multiprocessing Pool
    num_processes = mp.cpu_count()
    with mp.Pool(processes=num_processes) as pool:
        # Map the process_fdos function to each fdos in date_list
        # This will return a list of tuples: [(mc1, cv1, EMC1), (mc2, cv2, EMC2), ...]
        results = pool.map(process_fdos, date_list)
    
    # Filter out any None results due to errors
    results = [res for res in results if res is not None]
    
    # Unzip the list of tuples into three separate lists
    present_value_mc_list, present_value_cv_list, present_value_EMC_list = zip(*results)
    
    # Convert them to lists (from zip object)
    present_value_mc_list = list(present_value_mc_list)
    present_value_cv_list = list(present_value_cv_list)
    present_value_EMC_list = list(present_value_EMC_list)

    # Obtaining Realized Price 
    realized_price = pp.product_price()
    
    # Plotting the present values
    plt.figure(figsize=(12, 8))
    plt.plot(realized_price, label = 'Realized Price')
    plt.plot(date_list[:len(present_value_mc_list)], present_value_mc_list, label='Monte Carlo (MC)')
    plt.plot(date_list[:len(present_value_cv_list)], present_value_cv_list, label='Control Variate (CV)')
    plt.plot(date_list[:len(present_value_EMC_list)], present_value_EMC_list, label='Empirical Martingale Correction (EMC)')
    plt.xlabel('FDOS')
    plt.ylabel('Present Value')
    plt.title('Present Value Estimates Using Various Variance Reduction Techniques')
    plt.legend()
    plt.grid(True)
    file_path = r"C:\Users\Admin\PycharmProjects\Simulation Techniques in Finance\.venv\MH4518\WY\Graphs\VarianceReduction_nsims1000"
    plt.savefig(file_path)
    plt.show()
    

if __name__ == '__main__':
    main()




