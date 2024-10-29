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
    Simulation for 1 day and 1 simulation 

    Params:
        fdos : Specified first day of simulation, to be used in pricing 
        data : get data function 
    
    """
    try:
        log_returns_list = []
        for ticker in cs.ticker_list:
            log_returns = np.log(data[ticker]/data[ticker].shift(1))
            log_returns.dropna(inplace = True) # A series
            log_returns_list.append(log_returns)
        
    except Exception as e:
        raise Exception("Error at generating log return.")


    try: 
        cov_matrix = np.cov(np.array(log_returns_list))
        L = np.linalg.cholesky(cov_matrix)
        
    except Exception as e:
        raise Exception("Error at covariance matrix.")

    try:
        
        date_list = dates.get_list_dates(fdos, cs.final_fixing_date)
        s0_vector_fdos = data.loc[fdos]
        sim_window = dates.num_business_days(fdos, cs.final_fixing_date)
        
        sim_data = pd.DataFrame(np.zeros((sim_window, cs.num_ticker)), columns = cs.ticker_list)
        Zlist = np.random.normal(0, 1, (cs.num_ticker, sim_window))
        for t in range(sim_window): # for each timestep
            Z = Zlist[:, t] # for each time step, use a iid randomness 
            LZ = np.dot(L, Z.T)
            
            for i in range(cs.num_ticker):
                s0_vector_fdos[i] = s0_vector_fdos[i] * np.exp(cs.interest_rate * cs.dt - 0.5 * cov_matrix[i][i] * cs.dt + LZ[i])
                
                sim_data.loc[t, cs.ticker_list[i]] = s0_vector_fdos[i]

    

    except Exception as e:
        raise Exception("Error at Simulation")
    
    sim_data.index = date_list
    return sim_data


def multi_asset_gbm_n_sims(plot : bool, plotasset : bool, nsims, data, fdos) -> pd.DataFrame:
    """
    n simulations for 1 day 

    params:
        fdos: specified day for simulation, to be used in pricing 
        nsims: number of simulations to be carried out on this specified day
    
    """
    
    sim_data_combine = pd.DataFrame()
    realised_price = data.loc[cs.initial_fixing_date: cs.final_fixing_date]
    try: 
        for n in range(nsims):
            
            sim_data = multi_asset_gbm(data, fdos)
            sim_data_combine = pd.concat([sim_data_combine,sim_data], axis =1 )
        # formatting Dates 
        date_list = dates.get_list_dates(fdos, cs.final_fixing_date)
        if (len(sim_data) == len(date_list)):
            sim_data_combine.index = date_list
        else:
            print((f"The length of sim_data and dates is different: {len(sim_data)} and {len(date_list)}\n"))
    except Exception as e:
        raise Exception("Error at Simulation")


        
    if plot == True: 
        fig, ax = plt.subplots()
        
        realised_price.plot(ax=ax)
        sim_data.plot(ax=ax)
        
    if plotasset:
        # Select odd-numbered columns for Lonza paths
        lonza_path = sim_data_combine.iloc[:, ::2]
        # Select the realized price for Lonza (assuming it's the first column)
        realised_price_lonza = realised_price.iloc[:, 0]  # Adjust the index if necessary

        # Select even-numbered columns for Sika paths
        sika_path = sim_data_combine.iloc[:, 1::2]
        # Select the realized price for Sika (assuming it's the second column)
        realised_price_sika = realised_price.iloc[:, 1]  # Adjust the index if necessary

        # Plot Lonza paths and realized price
        fig, ax = plt.subplots()
        lonza_path.plot(ax=ax, alpha=0.4, legend=False)
        realised_price_lonza.plot(ax=ax, color='black', linewidth=2, label='Realised Price Lonza')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=[handles[-1]], labels=[labels[-1]])  # Only keep the last legend entry
        plt.title('Lonza Path under Multi Asset GBM')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.show()

        # Plot Sika paths and realized price
        fig, ax = plt.subplots()
        sika_path.plot(ax=ax, alpha=0.4, legend=False)
        realised_price_sika.plot(ax=ax, color='black', linewidth=2, label='Realised Price Sika')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=[handles[-1]], labels=[labels[-1]])  # Only keep the last legend entry
        plt.title('Sika Path under Multi Asset GBM')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.show()
    return sim_data_combine