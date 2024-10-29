import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yfin
import constants as cs
import Dates as dates 
import payoff as pf



def multi_asset_gbm(data: pd.DataFrame, fdos) -> pd.DataFrame:
    """
    Simulates multiple asset paths under the risk-neutral measure using GBM.

    Params:
        data : DataFrame containing historical price data for each asset.
        fdos : First date of simulation (datetime object).
    
    Returns:
        sim_data : DataFrame containing simulated asset prices.
    """
    import numpy as np
    import pandas as pd
    
    # Constants (assuming they are defined elsewhere)
    # For example:
    # cs.ticker_list = ['Asset1', 'Asset2']
    # cs.interest_rate = 0.01  # Risk-free rate
    # cs.dt = 1/252  # Time step in years (daily steps)
    # cs.final_fixing_date = datetime object
    # cs.num_ticker = len(cs.ticker_list)
    
    try:
        # Prepare the log returns DataFrame
        log_returns_df = pd.DataFrame()
        for ticker in cs.ticker_list:
            log_returns = np.log(data[ticker] / data[ticker].shift(1))
            log_returns_df[ticker] = log_returns
        log_returns_df.dropna(inplace=True)
        
    except Exception as e:
        raise Exception("Error at generating log returns.") from e

    try:
        # Compute the covariance matrix of log returns
        cov_matrix = log_returns_df.cov().values  # Shape: (num_assets, num_assets)
        # Compute the standard deviations (volatilities)
        vol_vector = np.sqrt(np.diag(cov_matrix))
        # Cholesky decomposition
        L = np.linalg.cholesky(cov_matrix)
        
    except Exception as e:
        raise Exception("Error at covariance matrix computation.") from e

    try:
        # Simulation parameters
        date_list = dates.get_list_dates(fdos, cs.final_fixing_date)
        s0_vector = data.loc[fdos, cs.ticker_list].values  # Initial prices as NumPy array
        sim_window = dates.num_business_days(fdos, cs.final_fixing_date)
        dt = cs.dt  # Time step in years
        
        # Initialize the simulated data DataFrame
        sim_data = pd.DataFrame(index=date_list, columns=cs.ticker_list)
        sim_data.iloc[0] = s0_vector  # Set initial prices
        
        # Precompute drift terms
        drift = (cs.interest_rate - 0.5 * vol_vector ** 2) * dt
        
        # Generate random variables for all time steps and assets
        Z = np.random.normal(size=(sim_window - 1, cs.num_ticker))
        # Generate correlated random variables
        epsilon = Z @ L.T * np.sqrt(dt)  # Shape: (sim_window - 1, num_assets)
        
        # Simulate asset prices
        S = np.zeros((sim_window, cs.num_ticker))
        S[0] = s0_vector  # Initial prices
        
        for t in range(1, sim_window):
            S[t] = S[t - 1] * np.exp(drift + epsilon[t - 1])
            sim_data.iloc[t] = S[t]
        
    except Exception as e:
        raise Exception("Error during simulation.") from e

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
        sim_data_combine (pd.DataFrame): DataFrame containing simulated asset prices for all simulations.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Assuming 'cs' and 'dates' are defined and accessible
    simulations = []
    
    # Get the list of dates for the simulation period
    date_list = dates.get_list_dates(fdos, cs.final_fixing_date)
    
    # Realized price from initial fixing date to final fixing date
    realised_price = data.loc[cs.initial_fixing_date: cs.final_fixing_date]
    
    try:
        for n in range(nsims):
            # Run the simulation
            sim_data = multi_asset_gbm(data, fdos)
            # Ensure sim_data has the correct index (dates)
            sim_data.index = date_list
            # Rename columns to include the simulation number
            sim_data.columns = [f"{col}_sim{n+1}" for col in sim_data.columns]
            simulations.append(sim_data)
    except Exception as e:
        raise Exception("Error during simulation") from e

    # Combine all simulations along the columns
    sim_data_combine = pd.concat(simulations, axis=1)

    # Ensure the index is set to date_list
    sim_data_combine.index = date_list

    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot the realized prices
        realised_price.plot(ax=ax, linewidth=2)
        # Plot the simulated paths
        sim_data_combine.plot(ax=ax, alpha=0.4, legend=False)
        plt.title('Simulated Paths with Realized Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.show()

    if plotasset:
        for asset in cs.ticker_list:
            # Select columns for the asset
            asset_columns = [col for col in sim_data_combine.columns if col.startswith(asset)]
            asset_paths = sim_data_combine[asset_columns]
            # Select the realized price for the asset
            realised_price_asset = realised_price[asset]  # Adjust the column name if necessary

            # Plot asset paths and realized price
            fig, ax = plt.subplots(figsize=(12, 6))
            asset_paths.plot(ax=ax, alpha=0.4, legend=False)
            realised_price_asset.plot(ax=ax, color='black', linewidth=2, label=f'Realized Price {asset}')
            ax.legend()
            plt.title(f'{asset} Paths under Multi Asset GBM')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
            plt.show()
            
    return sim_data_combine


