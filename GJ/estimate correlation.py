# Optionally, estimate correlation parameters between Lonza and Sika
import pandas as pd
import numpy as np

# Load historical price data for both assets
lonza_prices = pd.read_csv('/Users/guanjuwang/Desktop/TAN /MH4518/lonza.csv')  # Update with your file path
sika_prices = pd.read_csv('/Users/guanjuwang/Desktop/TAN /MH4518/sika.csv')    # Update with your file path

# Simulate some price data for testing
dates = pd.date_range(start='2022-01-01', periods=100, freq='B')
prices_lonza = pd.DataFrame({'Date': dates, 'Close_Lonza': np.linspace(100, 150, 100)})
prices_sika = pd.DataFrame({'Date': dates, 'Close_Sika': np.linspace(200, 250, 100)})

# Preprocess price data
lonza_prices['Date'] = pd.to_datetime(lonza_prices['Date'])
sika_prices['Date'] = pd.to_datetime(sika_prices['Date'])

# Merge the data on 'Date'
merged_prices = pd.merge(lonza_prices, sika_prices, on='Date', suffixes=('_Lonza', '_Sika'))

# Calculate daily returns
merged_prices['Return_Lonza'] = merged_prices['Close_Lonza'].pct_change()
merged_prices['Return_Sika'] = merged_prices['Close_Sika'].pct_change()

# Drop missing values
merged_prices.dropna(subset=['Return_Lonza', 'Return_Sika'], inplace=True)

# Calculate asset return correlation
rho_S_LONN_S_SIKA = merged_prices['Return_Lonza'].corr(merged_prices['Return_Sika'])
print(f"\nCorrelation between Lonza and Sika returns: {rho_S_LONN_S_SIKA}")

# Calculate variance process correlation using squared returns as proxies
merged_prices['Variance_Lonza'] = merged_prices['Return_Lonza'] ** 2
merged_prices['Variance_Sika'] = merged_prices['Return_Sika'] ** 2

rho_v_LONN_v_SIKA = merged_prices['Variance_Lonza'].corr(merged_prices['Variance_Sika'])
print(f"Correlation between Lonza and Sika variance proxies: {rho_v_LONN_v_SIKA}")