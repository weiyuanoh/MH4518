import requests
import json
import pandas as pd 
import numpy as np 

def get_product_price():
    api_url = 'https://derivative.credit-suisse.com/ch/ch/en/chart/producthistorical/instrumentID/483328'

    params = {
        'instrumentID': '483328',
        'fromDate': '2023-04-27',
        'toDate': '2024-07-30',
        'chartType': 'line',  # Example parameter
        # Include other parameters as needed
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Referer': 'https://derivative.credit-suisse.com/',  # Include if necessary
        'X-Requested-With': 'XMLHttpRequest',  # Indicates an AJAX request
    }

    cookies = {
        'Cookie' : 'JURISDICTION=ch; COUNTRY=ch; LANGUAGE=en; CFCLIENT_DERIVATIVE_4_0=""; CFID=4246488; CFTOKEN=35178305; CFGLOBALS=urltoken%3DCFID%23%3D4246488%26CFTOKEN%23%3D35178305%23lastvisit%3D%7Bts%20%272024%2D11%2D01%2007%3A36%3A06%27%7D%23hitcount%3D37%23timecreated%3D%7Bts%20%272024%2D10%2D28%2010%3A49%3A51%27%7D%23cftoken%3D35178305%23cfid%3D4246488%23'
    }


    response = requests.get( api_url, params= params, headers = headers, cookies= cookies )


    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Convert to DataFrame
        productprice = pd.DataFrame(data)

        # Process the DataFrame as needed
        productprice['date'] = pd.to_datetime(productprice['date'])
        productprice.sort_values('date', inplace=True)
        productprice.index = productprice['date']
        productprice = productprice['value'] * 10

    return productprice


# read interest functions 
def read_hist_rates():
    swiss_1_week = pd.read_csv(r"C:\Users\Espietsp\PycharmProjects\Simulation Techniques\.venv\MH4518\WY\Data\Switzerland 1-Week Bond Yield Historical Data (2).csv")
    swiss_1_month = pd.read_csv(r"C:\Users\Espietsp\PycharmProjects\Simulation Techniques\.venv\MH4518\WY\Data\Switzerland 1-Month Bond Yield Historical Data.csv")
    swiss_2_month = pd.read_csv(r"C:\Users\Espietsp\PycharmProjects\Simulation Techniques\.venv\MH4518\WY\Data\Switzerland 2-Month Bond Yield Historical Data.csv")
    swiss_6_month = pd.read_csv(r"C:\Users\Espietsp\PycharmProjects\Simulation Techniques\.venv\MH4518\WY\Data\Switzerland 6-Month Bond Yield Historical Data.csv")
    swiss_1_year = pd.read_csv(r"C:\Users\Espietsp\PycharmProjects\Simulation Techniques\.venv\MH4518\WY\Data\Switzerland 1-Year Bond Yield Historical Data.csv")
    combined = pd.concat([swiss_1_week["Price"], swiss_1_month["Price"], swiss_2_month["Price"], swiss_6_month["Price"], swiss_1_year["Price"]], axis = 1)
    return combined