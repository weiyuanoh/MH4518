import pandas as pd 

initial_fixing_date:pd.Timestamp = pd.Timestamp('2023-04-27').tz_localize(None)

final_fixing_date:pd.Timestamp = pd.Timestamp('2024-07-30').tz_localize(None)

data_enddate : pd.Timestamp = pd.Timestamp('2024-07-30').tz_localize(None)

data_startdate : pd.Timestamp = pd.Timestamp('2010-01-01').tz_localize(None)

sim_start_date : pd.Timestamp = initial_fixing_date + pd.offsets.BDay(1)

ticker_list: list = ['LONN.SW', 'SIKA.SW']

sim_ticker : list = ['LONN.SW_sim', 'SIKA.SW_sim']

num_ticker :int = len(ticker_list)

n_steps_per_year = 252

dt = 1/n_steps_per_year

window_len = (final_fixing_date - initial_fixing_date).days + 1

interest_rate = 0.018

n_sims = 100

initialS1 : float = 549.60

initialS2 : float = 240.40

barrierS1 : float = 329.76

barrierS2 : float = 144.24