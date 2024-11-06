import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yfin
import scipy
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

def getdata():
    data = yfin.download(['LONN.SW', 'SIKA.SW'], period= 'max')['Adj Close']
    data.index = data.index.tz_localize(None)
    data = data.loc[cs.initial_fixing_date: cs.final_fixing_date]
    return data 

data = getdata()

def read_hist_rates():
    swiss_1_week = pd.read_csv(r'C:\Users\Admin\PycharmProjects\Simulation Techniques in Finance\.venv\MH4518\WY\Data\Switzerland 1-Week Bond Yield Historical Data.csv', index_col = 0, parse_dates=True, infer_datetime_format=True)
    swiss_1_month = pd.read_csv(r'C:\Users\Admin\PycharmProjects\Simulation Techniques in Finance\.venv\MH4518\WY\Data\Switzerland 1-Month Bond Yield Historical Data.csv', index_col = 0 , parse_dates=True, infer_datetime_format=True)
    swiss_2_month = pd.read_csv(r'C:\Users\Admin\PycharmProjects\Simulation Techniques in Finance\.venv\MH4518\WY\Data\Switzerland 2-Month Bond Yield Historical Data.csv', index_col = 0 , parse_dates=True, infer_datetime_format=True)
    swiss_6_month = pd.read_csv(r'C:\Users\Admin\PycharmProjects\Simulation Techniques in Finance\.venv\MH4518\WY\Data\Switzerland 6-Month Bond Yield Historical Data.csv', index_col = 0 , parse_dates=True, infer_datetime_format=True)
    swiss_1_year = pd.read_csv(r'C:\Users\Admin\PycharmProjects\Simulation Techniques in Finance\.venv\MH4518\WY\Data\Switzerland 1-Year Bond Yield Historical Data.csv' , index_col = 0 , parse_dates=True, infer_datetime_format=True)
    combined = pd.concat([swiss_1_week[["Price"]], swiss_1_month[["Price"]], swiss_2_month[["Price"]], swiss_6_month[["Price"]], swiss_1_year[["Price"]]], axis = 1)
    combined.columns = ["1 week Price", "1 month Price", "2 month Price", "6 month Price", "1 Year Price"]
   
    return combined

df = read_hist_rates()
maturity_to_years = {
    '1 week Price': 1/52,
    '1 month Price': 1/12,
    '2 month Price': 2/12,
    '6 month Price': 6/12,
    '1 Year Price': 1
}
maturities = df.columns.tolist()
times_to_maturity = np.array([maturity_to_years[m] for m in maturities])

for col in df.columns:
    df[col] = df[col].apply(lambda x: np.exp(-x/100*maturity_to_years[col]))

def vasicek_bond_price(a, b, sigma, r0, T):
    B = (1 - np.exp(-a * T)) / a
    A = (b - (sigma**2) / (2 * a**2)) * (B - T) - (sigma**2) * B**2 / (4 * a)
    P = np.exp(A - B * r0)
    return P

from scipy.optimize import minimize
def calibration_objective(params, bond_maturities, bond_prices, r0):
    a, b, sigma = params
    # Ensure sigma is positive
    if sigma < 0:
        return np.inf
    model_prices = vasicek_bond_price(a, b, sigma, r0, bond_maturities)
    error = bond_prices - model_prices
    return np.sum(error**2)


def calibrate_vasicek(bond_maturities, bond_prices, r0):
    initial_guess = [0.1, 0.05, 0.01]  # [a, b, sigma]
    bounds = [
        (0.01, 1.0),   # a
        (0.01, 0.15),  # b
        (0.001, 0.1)    # sigma
    ]
    
    result = minimize(
        calibration_objective,
        initial_guess,
        args=(bond_maturities, bond_prices, r0),
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': True}
    )
    
    if result.success:
        calibrated_a, calibrated_b, calibrated_sigma = result.x
        calibrated_params = {
            'a': calibrated_a,
            'b': calibrated_b,
            'sigma': calibrated_sigma
        }
        print(f"Calibration successful: {calibrated_params}")
        return calibrated_params
    else:
        raise ValueError("Calibration failed: " + result.message)
    
calibrated_params = calibrate_vasicek(times_to_maturity, df.loc[cs.initial_fixing_date], 0.015)
def simulate_vasicek(a, b, sigma, r0, T, dt, N_simulations):
    """
    Simulate paths for the stochastic Vasicek interest rate model.
    
    Parameters:
    - a: Speed of mean reversion
    - b: Long-term mean level
    - sigma: Volatility of the interest rate
    - r0: Initial interest rate
    - T: Time horizon (years)
    - dt: Time step size
    - N_simulations: Number of simulation paths
    
    Returns:
    - time_grid: Array of time points
    - r_paths: Simulated interest rate paths (N_simulations x N_steps)
    """
    N_steps = int(T / dt)
    time_grid = np.linspace(0, T, N_steps)
    r_paths = np.zeros((N_simulations, N_steps))
    r_paths[:, 0] = r0
    
    dates = pd.date_range(cs.initial_fixing_date, cs.final_fixing_date)
    for t in range(1, N_steps):
        dr = a * (b - r_paths[:, t-1]) * dt + sigma * np.sqrt(dt) * np.random.randn(N_simulations)
        r_paths[:, t] = r_paths[:, t-1] + dr
    
    return time_grid, r_paths
a_calibrated = calibrated_params['a']
b_calibrated = calibrated_params['b']
sigma_r_calibrated = calibrated_params['sigma']

time_grid, r_paths = simulate_vasicek(a_calibrated, b_calibrated, sigma_r_calibrated, 0.015, (461/365), 1/365, 1000)


def plot_vasicek_matplotlib(time_grid, r_paths, num_paths=1000):
    """
    Plot a subset of Vasicek interest rate paths using Matplotlib.
    
    Parameters:
    - time_grid: Array of time points
    - r_paths: Simulated interest rate paths (N_simulations x N_steps)
    - num_paths: Number of paths to plot
    """
    plt.figure(figsize=(12, 6))
    
    # Randomly select paths to plot
    selected_indices = np.random.choice(r_paths.shape[0], size=num_paths, replace=False)
    
    for idx in selected_indices:
        plt.plot(time_grid, r_paths[idx], lw=0.5, alpha=0.6)
    
    plt.title('Stochastic Vasicek Interest Rate Paths')
    plt.xlabel('Time (Years)')
    plt.ylabel('Interest Rate')
    plt.grid(True)
    plt.show()

#Plot Vasicek Interest Rate Paths
plot_vasicek_matplotlib(time_grid, r_paths, num_paths=1000)


from datetime import datetime
from dateutil.relativedelta import relativedelta

# Dates Formatting
initial_date = cs.initial_fixing_date
new_date = initial_date + relativedelta(years=1, months=3, days = 3)
r_paths = pd.DataFrame(r_paths).T
dates_index= pd.date_range(cs.initial_fixing_date, new_date)
r_paths.index = dates_index


def multi_asset_gbm_div_interest(data: pd.DataFrame, fdos, nsims: int, r_paths) -> pd.DataFrame:
    """
    Simulates multiple asset paths under the risk-neutral measure using GBM for multiple simulations.
    """

    
    try:
        # Prepare the log returns DataFrame
        log_returns_df = pd.DataFrame()
        for ticker in cs.ticker_list:
            log_returns = np.log(data[ticker] / data[ticker].shift(1))
            log_returns_df[ticker] = log_returns
        log_returns_df.dropna(inplace=True)
        
        # Compute the covariance matrix of log returns
        cov_matrix = log_returns_df.cov().values
        vol_vector = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(vol_vector, vol_vector)
        L = np.linalg.cholesky(corr_matrix)
        
    except Exception as e:
        print("Error at covariance matrix computation:", e)
        raise
    
    try:
        # Simulation parameters
        date_list = dates.get_list_dates(fdos, cs.final_fixing_date)
        num_steps = len(date_list)  # Number of time steps
        dt = cs.dt  # Time step in years
        
        num_assets = cs.num_ticker  # Number of assets
        
        s0_vector = data.loc[fdos, cs.ticker_list].values  # Initial prices as NumPy array
        
        # Print lengths
        print(f"Length of date_list: {len(date_list)}")
        print(f"Number of time steps (num_steps): {num_steps}")
        
        # Precompute drift terms
        r = np.mean(r_paths.loc[fdos])
        drift = (r - 0.5 * vol_vector ** 2) * dt  # Shape: (num_assets,)
        
        # Generate random variables
        Z = np.random.normal(size=(num_steps - 1, nsims, num_assets))  # Shape: (num_steps-1, nsims, num_assets)
        Z_reshaped = Z.reshape(-1, num_assets)  # Shape: ((num_steps - 1) * nsims, num_assets)
        epsilon = Z_reshaped @ L.T  # Shape: ((num_steps - 1) * nsims, num_assets)
        epsilon = epsilon.reshape(num_steps - 1, nsims, num_assets)  # Shape: (num_steps - 1, nsims, num_assets)
        epsilon = epsilon * np.sqrt(dt)  # Scale by sqrt(dt)
        
        # Initialize array to hold simulated prices
        S = np.zeros((num_steps, nsims, num_assets))  # Shape: (num_steps, nsims, num_assets)
        S[0, :, :] = s0_vector  # Set initial prices for all simulations

# Initialize Dividend Search Dictionaries
        lonza_dividend_search = {
            pd.Timestamp('2023-05-11'): 3.50,
            pd.Timestamp('2024-05-15'): 4.00
        }
        
        sika_dividend_search = {
            pd.Timestamp('2024-04-03'): 3.30
        }
        
        # Create Date-to-Step Mapping
        date_to_step = {date: idx for idx, date in enumerate(date_list)}
        
        # Function to find simulation step for a dividend date
        def find_simulation_step(dividend_date, date_list, date_to_step):
            """
            Finds the simulation step (index) for a given dividend date.
            If the exact date isn't present, adjusts to the next available simulation date.
            """
            if dividend_date in date_to_step:
                return date_to_step[dividend_date]
            else:
                # Find the next date in date_list after dividend_date
                future_dates = [d for d in date_list if d > dividend_date]
                if future_dates:
                    nearest_date = future_dates[0]
                    return date_to_step[nearest_date]
                else:
                    # If no future date is found, return None or handle accordingly
                    return None
        
        # Convert Lonza Dividend Dates
        dividend_steps_lonza = {}
        for date, amount in lonza_dividend_search.items():
            step = find_simulation_step(date, date_list, date_to_step)
            if step is not None:
                dividend_steps_lonza[step] = amount
            else:
                print(f"Dividend date {date.date()} for Lonza not found in simulation dates.")
        
        # Convert Sika Dividend Dates
        dividend_steps_sika = {}
        for date, amount in sika_dividend_search.items():
            step = find_simulation_step(date, date_list, date_to_step)
            if step is not None:
                dividend_steps_sika[step] = amount
            else:
                print(f"Dividend date {date.date()} for Sika not found in simulation dates.")
        
        # Simulate asset prices
        for t in range(1, num_steps):
            S[t] = S[t - 1] * np.exp(
                drift + vol_vector * epsilon[t - 1]
            )
            if t in dividend_steps_lonza:
                asset_idx = cs.ticker_list.index('LONN.SW')  # Get the index of Lonza in the asset list
                dividend_amount = dividend_steps_lonza[t]
                S[t, :, asset_idx] -= dividend_amount  # Subtract dividend

            if t in dividend_steps_sika:
                asset_idx = cs.ticker_list.index('SIKA.SW')  # Get the index of Sika in the asset list
                dividend_amount = dividend_steps_sika[t]
                S[t, :, asset_idx] -= dividend_amount  # Subtract dividend
        
        # Transpose S to shape (num_steps, num_assets, nsims)
        S_transposed = S.transpose(0, 2, 1)  # Swap axes 1 and 2
        
        # Reshape S_transposed to (num_steps, num_assets * nsims)
        S_reshaped = S_transposed.reshape(num_steps, num_assets * nsims)
        
        # Build MultiIndex columns
        assets = cs.ticker_list
        simulations = [f'sim{n+1}' for n in range(nsims)]
        arrays = [[], []]
        for asset in assets:
            for sim in simulations:
                arrays[0].append(asset)
                arrays[1].append(sim)
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['Asset', 'Simulation'])
        
        # Create DataFrame
        sim_data = pd.DataFrame(S_reshaped, index=date_list)
        sim_data.columns = index
        
    except Exception as e:
        print("Error during simulation:", e)
        raise
        
    return sim_data 

def multi_asset_gbm_n_sims(plot: bool, plotasset: bool, nsims: int, data: pd.DataFrame, fdos) -> pd.DataFrame:
    """
    Simulate multiple asset paths under the GBM model for 'nsims' simulations starting from 'fdos'.

    Params:
        plot (bool): Whether to plot the combined simulations.
        plotasset (bool): Whether to plot the simulations for each asset separately.
        nsims (int): Number of simulations to run.
        data (pd.DataFrame): Historical data for assets.
        fdos: First date of simulation.

    Returns:
        sim_data: DataFrame containing simulated asset prices for all simulations.
                  Columns are MultiIndex with levels ('Asset', 'Simulation')
    """

    # Get the list of dates for the simulation period
    date_list = dates.get_list_dates(fdos, cs.final_fixing_date)
    
    # Realized price from initial fixing date to final fixing date
    realised_price = data.loc[cs.initial_fixing_date: cs.final_fixing_date]
    
    try:
        # Run the simulation
        sim_data = multi_asset_gbm_div_interest(data, fdos, nsims, r_paths)
        sim_data.index = date_list
        
    except Exception as e:
        raise Exception("Error during simulation") from e

    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot the realized prices
        realised_price.plot(ax=ax, linewidth=2)
        # Plot the simulated paths
        for asset in cs.ticker_list:
            asset_columns = sim_data[asset]
            asset_columns.plot(ax=ax, alpha=0.4, legend=False)
        plt.title('Simulated Paths with Realized Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.show()

    if plotasset:
        for asset in cs.ticker_list:
            # Select columns for the asset
            asset_columns = sim_data[asset]
            # Select the realized price for the asset
            realised_price_asset = realised_price[asset]  # Adjust the column name if necessary

            # Plot asset paths and realized price
            fig, ax = plt.subplots(figsize=(12, 6))
            asset_columns.plot(ax=ax, alpha=0.4, legend=False)
            realised_price_asset.plot(ax=ax, color='black', linewidth=2, label=f'Realized Price {asset}')
            ax.legend()
            plt.title(f'{asset} Paths under Multi Asset GBM')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
            plt.show()
            
    return sim_data


params_product = {
    'Denomination': 1000,
    'Coupon_Rate': (0.08 / 4) ,  # Quarterly coupon payments
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

        # Load or access necessary data
        # If 'data', 'params_product', 'cs', and 'dates' are global, you can access them directly
        # Alternatively, pass them as arguments or load them within the function

        # Run the simulation
        sim_T = multi_asset_gbm_n_sims(plot = False,
                                       plotasset=False,
                                       nsims = cs.n_sims,
                                       data = data, 
                                       fdos = fdos)
        
        # Extract asset paths
        lonza_path = sim_T['LONN.SW']
        sika_path = sim_T['SIKA.SW']
        # Calculate payoffs
        payoff_T = pf.payoff(lonza_path, sika_path, params_product, fdos)
        # Compute expected payoff
        expected_payoff = np.mean(payoff_T)
        # Discount to present value (assuming present value as of fdos)
        r = np.mean(r_paths.loc[fdos])
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

    n=fig, ax = plt.subplots(figsize=(12, 6))
    present_value_df = pd.DataFrame({'Avg Payoff': present_value_list}, index=T)
    present_value_df.plot(ax = ax)



    productprice.plot(ax =ax)
    plt.show()


