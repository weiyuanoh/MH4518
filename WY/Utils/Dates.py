import pandas as pd 
import pandas_market_calendars as cal 
import MH4518.WY.Utils.constants as cs

def num_business_days(startdate, enddate):
    """ 
    given the start date and end date, the number of business days in the SWISS exchange is calculated 
    
    all parameters are timezone naive 
    Parameters:
        startdate (pd.timestamp) : the start date to be used in calculation
        enddate (pd.timestamp): the end date to be used in calculation 
        
    Returns:
        (int) : number of business days in timezone neutral using timezone neutral dates    
    """
    #converting dates to timezone naive
    startdate = pd.Timestamp(startdate).tz_localize(None)
    enddate = pd.Timestamp(enddate).tz_localize(None)
    swiss_cal = cal.get_calendar("SIX")
    n_busi_days = len(swiss_cal.valid_days(startdate, enddate))
    return n_busi_days
 

def get_early_observation_dates(initialdate, enddate):
    """
    Given start date and end date

    Returns:
        list of early observation dates between start date and end date
    """
    # Define the SIX exchange calendar
    six_cal = cal.get_calendar('SIX')

    # Calculate the first observation date
    first_observation_date = initialdate + pd.DateOffset(months=6)

    # Generate the date range with quarterly frequency
    redemption_dates = pd.date_range(start=first_observation_date, end=enddate, freq='Q')

    # Filter the dates to include only trading days
    trading_days = six_cal.valid_days(start_date=first_observation_date, end_date=enddate)
    redemption_dates = [date for date in redemption_dates if date in trading_days]

    # Make dates timezone naive
    redemption_dates = [date.tz_localize(None) for date in redemption_dates]

    return redemption_dates


def add_business_days(date: pd.Timestamp):
    # Get the valid trading days starting from the next day
    six_cal = cal.get_calendar('SIX')
    valid_days = six_cal.valid_days(start_date=date + pd.tseries.offsets.BusinessDay(1), end_date=date + pd.Timedelta(days=10))
    
    # Return the first valid day
    return valid_days[0]

def get_list_dates(startdate: pd.Timestamp, enddate: pd.Timestamp):
    """
    Parameters:
        startdate : initial date to calculate date range 
        enddate : final date to calculate date range 

    Returns:
        list of dates in between these 2 dates inclusive according to the SIX exchange calendar
    """
    # Define the SIX exchange calendar
    six_cal = cal.get_calendar('SIX')

    # Get the valid trading days between the start and end dates
    trading_days = six_cal.valid_days(start_date=startdate, end_date=enddate)

    # Make dates timezone naive
    date_list = [date.tz_localize(None) for date in trading_days]

    return date_list


def business_days_in_quarter(start_date: pd.Timestamp, end_date: pd.Timestamp):
    # Define the SIX exchange calendar
    six_cal = cal.get_calendar('SIX')

    # Get the valid trading days between the start and end dates
    trading_days = six_cal.valid_days(start_date=start_date, end_date=end_date)

    # Make dates timezone naive
    trading_days = trading_days.tz_localize(None)

    # Count the number of business days
    num_business_days = len(trading_days)

    return num_business_days