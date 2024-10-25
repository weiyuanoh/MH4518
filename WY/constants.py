import pandas as pd 

initial_fixing_date:pd.Timestamp = pd.Timestamp('2023-04-27')

final_fixing_date:pd.Timestamp = pd.Timestamp('2024-07-30')

data_enddate : pd.Timestamp = pd.Timestamp('2024-07-30')

data_startdate : pd.Timestamp = pd.Timestamp('2010-01-01')

sim_start_date : pd.Timestamp = initial_fixing_date + pd.offsets.BDay(1)