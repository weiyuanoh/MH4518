import requests
import json
import pandas as pd 
import numpy as np 

def get_product_price():
    api_url = 'https://derivative.credit-suisse.com/ch/ch/en/chart/producthistorical/instrumentID/483328'

    params = {
        'instrumentID': '483328',
        'fromDate': '2023-04-27',
        'toDate': '2024-07-30',  # Ensure this date is not in the future relative to today's date
        'chartType': 'line',  # Example parameter
        # Include other parameters as needed
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Referer': 'https://derivative.credit-suisse.com/ch/ch/en/detail/autocallable-brc-lonza-sika-8-00-p-a/CH1253871557/125387155',  # Include if necessary
        'X-Requested-With': 'XMLHttpRequest',  # Indicates an AJAX request
    }

    cookies = {
        'Cookie' : 'JURISDICTION=ch; COUNTRY=ch; LANGUAGE=en; CFCLIENT_DERIVATIVE_4_0=""; CFID=4342638; CFTOKEN=47971302; CFGLOBALS=urltoken%3DCFID%23%3D4342638%26CFTOKEN%23%3D47971302%23lastvisit%3D%7Bts%20%272024%2D11%2D05%2006%3A47%3A56%27%7D%23hitcount%3D55%23timecreated%3D%7Bts%20%272024%2D10%2D28%2010%3A49%3A51%27%7D%23cftoken%3D47971302%23cfid%3D4342638%23'
    }

    try:
        response = requests.get(api_url, params=params, headers=headers, cookies=cookies, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

    if response.status_code == 200:
        try:
            # Attempt to parse JSON
            data = response.json()
        except json.JSONDecodeError as e:
            print("JSON decoding failed. Response content:")
            print(response.text)  # Print the response text for debugging
            return None

        # Convert to DataFrame
        if isinstance(data, list) or isinstance(data, dict):
            productprice = pd.DataFrame(data)
        else:
            print("Unexpected JSON structure:", type(data))
            return None
    
            print("Expected columns 'date' and 'value' not found in the data.")
            return None

        return productprice
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        print("Response content:")
        print(response.text)  # Print the response text for debugging
        return None

# read interest functions 
def read_hist_rates():
    swiss_1_week = pd.read_csv(r"C:\Users\Espietsp\PycharmProjects\Simulation Techniques\.venv\MH4518\WY\Data\Switzerland 1-Week Bond Yield Historical Data (2).csv")
    swiss_1_month = pd.read_csv(r"C:\Users\Espietsp\PycharmProjects\Simulation Techniques\.venv\MH4518\WY\Data\Switzerland 1-Month Bond Yield Historical Data.csv")
    swiss_2_month = pd.read_csv(r"C:\Users\Espietsp\PycharmProjects\Simulation Techniques\.venv\MH4518\WY\Data\Switzerland 2-Month Bond Yield Historical Data.csv")
    swiss_6_month = pd.read_csv(r"C:\Users\Espietsp\PycharmProjects\Simulation Techniques\.venv\MH4518\WY\Data\Switzerland 6-Month Bond Yield Historical Data.csv")
    swiss_1_year = pd.read_csv(r"C:\Users\Espietsp\PycharmProjects\Simulation Techniques\.venv\MH4518\WY\Data\Switzerland 1-Year Bond Yield Historical Data.csv")
    combined = pd.concat([swiss_1_week["Price"], swiss_1_month["Price"], swiss_2_month["Price"], swiss_6_month["Price"], swiss_1_year["Price"]], axis = 1)
    return combined

def product_price():
    productprice = pd.read_json(r'C:\Users\Espietsp\PycharmProjects\Simulation Techniques\.venv\MH4518\WY\Data\ProductPrice.json')
    productprice['date'] = pd.to_datetime(productprice['date'])
    productprice.sort_values('date', inplace=True)
    productprice.set_index('date', inplace=True)
    productprice = productprice['value'] * 10
    return productprice